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
	"context"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/cespare/xxhash/v2"
	"lukechampine.com/blake3"
	"sigs.k8s.io/controller-runtime/pkg/log"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requesthandling"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

// hashPrompt divides the prompt into blocks and calculates a prefix cache hash for each block.
// The first block hash includes the model name and cache salt (if provided).
// For subsequent blocks, the hash is calculated as: hash(block i content, hash(i-1)).
func hashPrompt(ctx context.Context, request *scheduling.InferenceRequest, blockSizeTokens int, maxPrefixBlocks int) []blockHash {
	loggerDebug := log.FromContext(ctx).V(logutil.DEBUG)
	if request == nil || request.Body == nil {
		loggerDebug.Info("Request or request data is nil, skipping hashing")
		return nil
	}

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
	for i := 0; i+cacheBlockSizeChars <= len(userInput); i += cacheBlockSizeChars {
		h.Reset()
		_, _ = h.Write(userInput[i : i+cacheBlockSizeChars])
		_, _ = h.Write(toBytes(prevBlockHash))
		res = append(res, blockHash(h.Sum64()))

		prevBlockHash = res[len(res)-1]
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
		// Deep copy messages to avoid modifying the original request object.
		// This is important for ensuring that subsequent plugins in the chain
		// see the original, unmodified request.
		originalMessages := request.Body.ChatCompletions.Messages
		clonedMessages, _ := DeepCloneMessages(originalMessages)
		for i := range clonedMessages {
			if len(clonedMessages[i].Content.Structured) > 0 {

				for j := range clonedMessages[i].Content.Structured {
					// Use a pointer to modify the actual item in the slice
					block := &clonedMessages[i].Content.Structured[j]

					if block.Type == "image_url" && block.ImageURL.Url != "" {
						// Hash the image URL using BLAKE3
						hash := blake3.Sum256([]byte(block.ImageURL.Url))

						// Encode to a readable, URL-safe hexadecimal string
						block.ImageURL.Url = hex.EncodeToString(hash[:])
					}
				}
			}
		}
		return json.Marshal(clonedMessages)

	case request.Body.Completions != nil:
		return []byte(request.Body.Completions.Prompt.PlainText()), nil

	case request.Body.Embeddings != nil:
		// Handle embeddings API - marshal input for cache key generation
		return json.Marshal(request.Body.Embeddings.Input)

	default:
		return nil, errors.New("invalid request body: no recognized API format found")
	}
}

func DeepCloneMessages(original []requesthandling.Message) ([]requesthandling.Message, error) {
	if original == nil {
		return nil, nil
	}

	// 1. Marshal the original slice to JSON bytes
	bytes, err := json.Marshal(original)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal messages: %w", err)
	}

	// 2. Unmarshal back into a new slice
	var cloned []requesthandling.Message
	if err := json.Unmarshal(bytes, &cloned); err != nil {
		return nil, fmt.Errorf("failed to unmarshal messages: %w", err)
	}

	return cloned, nil
}
