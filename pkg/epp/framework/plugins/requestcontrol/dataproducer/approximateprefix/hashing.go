/*
Copyright 2026 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package approximateprefix

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"strings"

	"github.com/cespare/xxhash/v2"
	"sigs.k8s.io/controller-runtime/pkg/log"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

type ContentBlock struct {
	TokenIDs  []string `json:"TokenIDs"`
	ExtraHash string   `json:"ExtraHash"`
	Size      int      `json:"Size"`
}

// hashPrompt divides the prompt into blocks and calculates a prefix cache hash for each block.
// The first block hash includes the model name and cache salt (if provided).
// For subsequent blocks, the hash is calculated as: hash(block i content, hash(i-1)).
func hashPrompt(ctx context.Context, request *scheduling.InferenceRequest, blockSizeTokens int, maxPrefixBlocks int) []blockHash {
	loggerDebug := log.FromContext(ctx).V(logutil.DEBUG)
	if request == nil || request.Body == nil {
		loggerDebug.Info("Request or request data is nil, skipping hashing")
		return nil
	}
	if request.Body.ChatCompletions == nil {

		userInput, err := getUserInputBytes(request)
		if err != nil {
			loggerDebug.Error(err, "Failed to get user input bytes")
			return nil
		}

		// convert block size from tokens to characters
		cacheBlockSizeChars := blockSizeTokens * averageCharactersPerToken

		if cacheBlockSizeChars <= 0 {
			loggerDebug.Info("Skipping prefix hashing: block size in characters must be positive",
				"blockSizeTokens", blockSizeTokens,
				"cacheBlockSizeChars", cacheBlockSizeChars)
			return nil
		}

		if len(userInput) < cacheBlockSizeChars {
			loggerDebug.Info("Request body too small for prefix cache", "size", len(userInput), "block size in chars", cacheBlockSizeChars)
			return nil
		}

		if len(userInput) > cacheBlockSizeChars*maxPrefixBlocks {
			loggerDebug.Info("Truncating input", "size", len(userInput), "max prefix blocks", maxPrefixBlocks, "block size in chars", cacheBlockSizeChars)
			userInput = userInput[:maxPrefixBlocks*cacheBlockSizeChars]
		}

		// Split the body into blocks of size cacheBlockSizeChars.
		res := make([]blockHash, 0, len(userInput)/cacheBlockSizeChars)

		h := xxhash.New()
		// Different models should have different hashes even with the same body.
		_, _ = h.Write([]byte(request.TargetModel))
		if cacheSalt := request.Body.CacheSalt(); cacheSalt != "" {
			_, _ = h.Write([]byte(cacheSalt))
		}

		prevBlockHash := blockHash(h.Sum64())
		i := 0
		for ; i+cacheBlockSizeChars <= len(userInput); i += cacheBlockSizeChars {
			h.Reset()
			_, _ = h.Write(userInput[i : i+cacheBlockSizeChars])
			_, _ = h.Write(toBytes(prevBlockHash))
			res = append(res, blockHash(h.Sum64()))

			prevBlockHash = res[len(res)-1]
		}

		// 2. Process any remaining bytes as a partial block
		if i < len(userInput) {
			h.Reset()

			_, _ = h.Write(userInput[i:])
			_, _ = h.Write(toBytes(prevBlockHash))
			res = append(res, blockHash(h.Sum64()))
		}

		return res
	}

	h := xxhash.New()
	// Different models should have different hashes even with the same body.
	_, _ = h.Write([]byte(request.TargetModel))
	if cacheSalt := request.Body.CacheSalt(); cacheSalt != "" {
		_, _ = h.Write([]byte(cacheSalt))
	}

	prevBlockHash := blockHash(h.Sum64())

	contentBlocks := GetContentBlocks(request, blockSizeTokens)
	res := make([]blockHash, 0, len(contentBlocks))

	for _, contentBlock := range contentBlocks {
		h.Reset()

		// 1. Serialize the entire ContentBlock into bytes
		blockBytes, err := json.Marshal(contentBlock)
		if err != nil {
			// Handle this error according to your app's needs
			panic(err)
		}

		// 2. Hash the entire serialized block
		_, _ = h.Write(blockBytes)

		// 3. Chain the previous block's hash to maintain the sequence
		_, _ = h.Write(toBytes(prevBlockHash))

		// 4. Calculate the current hash and append to results
		currentHash := blockHash(h.Sum64())
		res = append(res, currentHash)

		// 5. Update prevBlockHash for the next iteration
		prevBlockHash = currentHash
	}
	return res
}

func toBytes(i blockHash) []byte {
	bytes := make([]byte, 8)
	binary.LittleEndian.PutUint64(bytes, uint64(i))
	return bytes
}

func getUserInputBytes(request *scheduling.InferenceRequest) ([]byte, error) {
	switch {
	case request.Body.Conversations != nil:
		return json.Marshal(request.Body.Conversations.Items)

	case request.Body.Responses != nil:
		var combined []map[string]interface{}
		if request.Body.Responses.Instructions != nil {
			combined = append(combined, map[string]interface{}{"instructions": request.Body.Responses.Instructions})
		}
		if request.Body.Responses.Tools != nil {
			combined = append(combined, map[string]interface{}{"tools": request.Body.Responses.Tools})
		}
		combined = append(combined, map[string]interface{}{"input": request.Body.Responses.Input})
		return json.Marshal(combined)

	case request.Body.ChatCompletions != nil:
		return json.Marshal(request.Body.ChatCompletions.Messages)

	case request.Body.Completions != nil:
		return []byte(request.Body.Completions.Prompt.PlainText()), nil

	case request.Body.Embeddings != nil:
		// Handle embeddings API - marshal input for cache key generation
		return json.Marshal(request.Body.Embeddings.Input)

	default:
		return nil, errors.New("invalid request body: no recognized API format found")
	}
}

func GetContentBlocks(request *scheduling.InferenceRequest, blockSizeTokens int) []ContentBlock {
	if request == nil || request.Body == nil || request.Body.ChatCompletions == nil {
		return nil
	}

	messages := request.Body.ChatCompletions.Messages
	var allTokens []string
	var tokenImageHashes []string

	for _, msg := range messages {
		if msg.Content.Raw != "" {
			text := msg.Content.Raw
			for i := 0; i < len(text); i += 4 {
				end := i + 4
				if end > len(text) {
					end = len(text)
				}
				allTokens = append(allTokens, text[i:end])
				tokenImageHashes = append(tokenImageHashes, "")
			}
		} else if len(msg.Content.Structured) > 0 {
			for _, block := range msg.Content.Structured {
				switch block.Type {
				case "text":
					text := block.Text
					for i := 0; i < len(text); i += averageCharactersPerToken {
						end := i + averageCharactersPerToken
						if end > len(text) {
							end = len(text)
						}
						allTokens = append(allTokens, text[i:end])
						tokenImageHashes = append(tokenImageHashes, "")
					}
				case "image_url":
					url := block.ImageURL.Url
					var placeholders int
					if strings.HasPrefix(url, "data:image/") && strings.Contains(url, "base64,") {
						idx := strings.Index(url, "base64,")
						base64Data := url[idx+7:]
						decoded, err := base64.StdEncoding.DecodeString(base64Data)
						if err == nil {
							config, _, err := image.DecodeConfig(bytes.NewReader(decoded))
							if err == nil {
								placeholders = ((config.Width * config.Height) / (28 * 28)) + 2
							}
						}
					}
					h := xxhash.New()
					_, _ = h.Write([]byte(url))
					imgHash := fmt.Sprintf("%x", h.Sum64())

					for i := 0; i < placeholders; i++ {
						allTokens = append(allTokens, "P")
						tokenImageHashes = append(tokenImageHashes, imgHash)
					}
				}
			}
		}
	}

	var res []ContentBlock
	for i := 0; i < len(allTokens); i += blockSizeTokens {
		end := i + blockSizeTokens
		if end > len(allTokens) {
			end = len(allTokens)
		}

		blockTokens := allTokens[i:end]
		var extraHash string
		for j := i; j < end; j++ {
			if tokenImageHashes[j] != "" {
				extraHash += tokenImageHashes[j]
				break
			}
		}

		res = append(res, ContentBlock{
			TokenIDs:  blockTokens,
			ExtraHash: extraHash,
			Size:      blockSizeTokens,
		})
	}

	return res
}
