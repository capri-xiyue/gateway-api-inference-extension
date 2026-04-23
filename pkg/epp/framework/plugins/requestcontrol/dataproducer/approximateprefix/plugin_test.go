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
	"fmt"
	"math/rand"
	"strings"
	"testing"

	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	fwkrh "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requesthandling"
	fwksched "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	attrprefix "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/attribute/prefix"
)

func TestPrepareRequestData(t *testing.T) {
	config := config{
		BlockSizeTokens:        1,
		MaxPrefixBlocksToMatch: defaultMaxPrefixBlocks,
		LRUCapacityPerServer:   defaultLRUCapacityPerServer,
	}
	// Test the "initialize if nil" pattern
	p, err := newPrepareData(context.Background(), config, nil)
	assert.NoError(t, err)
	assert.NotNil(t, p.PluginState())

	endpoint1 := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}, fwkdl.NewMetrics(), fwkdl.NewAttributes())
	endpoint2 := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}}, fwkdl.NewMetrics(), fwkdl.NewAttributes())
	endpoints := []fwksched.Endpoint{endpoint1, endpoint2}

	// First request to populate cache.
	req1 := &fwksched.InferenceRequest{
		RequestId:   uuid.NewString(),
		TargetModel: "test-model1",
		Body: &fwkrh.InferenceRequestBody{
			Completions: &fwkrh.CompletionsRequest{
				Prompt: fwkrh.Prompt{Raw: "aaaabbbb"},
			},
		},
	}

	// We need to simulate the PreRequest logic since PrepareRequestData only reads from the indexer.
	// But first let's see if PrepareRequestData correctly handles an empty indexer.
	err = p.PrepareRequestData(context.Background(), req1, endpoints)
	assert.NoError(t, err)

	// Verify state was written to PluginState
	state, err := plugin.ReadPluginStateKey[*SchedulingContextState](p.PluginState(), req1.RequestId, plugin.StateKey(ApproxPrefixCachePluginType))
	assert.NoError(t, err)
	assert.NotNil(t, state)
	assert.Equal(t, 2, len(state.PrefixHashes)) // "aaaabbbb" with blockSize 4 (1 token * 4 chars) -> 2 blocks

	// Verify pod match info was set (should be 0 match since indexer is empty)
	for _, ep := range endpoints {
		info, ok := ep.Get(attrprefix.PrefixCacheMatchInfoKey)
		assert.True(t, ok)
		prefixInfo := info.(*attrprefix.PrefixCacheMatchInfo)
		assert.Equal(t, 0, prefixInfo.MatchBlocks())
		assert.Equal(t, 2, prefixInfo.TotalBlocks())
	}
}

func TestPreRequest(t *testing.T) {
	config := config{
		BlockSizeTokens:        1,
		MaxPrefixBlocksToMatch: defaultMaxPrefixBlocks,
		LRUCapacityPerServer:   defaultLRUCapacityPerServer,
	}
	p, _ := newPrepareData(context.Background(), config, nil)

	endpoint1 := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod1", Namespace: "default"}}, fwkdl.NewMetrics(), fwkdl.NewAttributes())
	req1 := &fwksched.InferenceRequest{
		RequestId:   uuid.NewString(),
		TargetModel: "test-model1",
		Body: &fwkrh.InferenceRequestBody{
			Completions: &fwkrh.CompletionsRequest{
				Prompt: fwkrh.Prompt{Raw: "aaaabbbb"},
			},
		},
	}

	// 1. Prepare data (this saves state)
	_ = p.PrepareRequestData(context.Background(), req1, []fwksched.Endpoint{endpoint1})

	// 2. Simulate scheduling result
	res := &fwksched.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"default": {
				TargetEndpoints: []fwksched.Endpoint{endpoint1},
			},
		},
	}

	// 3. Call PreRequest
	p.PreRequest(context.Background(), req1, res)

	// Wait for async update
	p.wg.Wait()

	// 4. Verify indexer was updated
	hashes := hashPrompt(context.Background(), req1, 4, defaultMaxPrefixBlocks)
	for _, hash := range hashes {
		pods := p.indexer().Get(hash)
		assert.Contains(t, pods, ServerID(endpoint1.GetMetadata().NamespacedName))
	}
}

func TestPrepareDataValidation(t *testing.T) {
	validConfigs := []config{{
		AutoTune:        false,
		BlockSizeTokens: 1,
	}, {
		AutoTune:        false,
		BlockSize:       1,
		BlockSizeTokens: 1,
	}, {
		AutoTune:        true,
		BlockSizeTokens: 0,
	}}
	invalidConfigs := []config{{
		AutoTune:  false,
		BlockSize: 1,
	}, {
		AutoTune:        false,
		BlockSizeTokens: 0,
	}}

	for _, config := range validConfigs {
		_, err := newPrepareData(context.Background(), config, nil)
		assert.NoError(t, err)
	}

	for _, config := range invalidConfigs {
		_, err := newPrepareData(context.Background(), config, nil)
		assert.Error(t, err)
	}
}

func TestPrefixPluginCompletion(t *testing.T) {
	config := config{
		BlockSizeTokens:        1,
		MaxPrefixBlocksToMatch: defaultMaxPrefixBlocks,
		LRUCapacityPerServer:   defaultLRUCapacityPerServer,
	}
	p, _ := newPrepareData(context.Background(), config, nil)

	endpoint1 := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}, fwkdl.NewMetrics(), fwkdl.NewAttributes())
	endpoint2 := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}}, fwkdl.NewMetrics(), fwkdl.NewAttributes())
	endpoint3 := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod3"}}, fwkdl.NewMetrics(), fwkdl.NewAttributes())
	endpoints := []fwksched.Endpoint{endpoint1, endpoint2, endpoint3}

	// First request.
	req1 := &fwksched.InferenceRequest{
		RequestId:   uuid.NewString(),
		TargetModel: "test-model1",
		Body: &fwkrh.InferenceRequestBody{
			Completions: &fwkrh.CompletionsRequest{
				Prompt: fwkrh.Prompt{Raw: "aaaaaa"},
			},
		},
	}
	_ = p.PrepareRequestData(context.Background(), req1, endpoints)
	state, _ := plugin.ReadPluginStateKey[*SchedulingContextState](p.PluginState(), req1.RequestId, plugin.StateKey(ApproxPrefixCachePluginType))
	// Input size is 6, block size is 4, so 1 body block. Total hashes = 1 (model only is not a block)
	assert.Equal(t, 2, len(state.PrefixHashes))

	// Simulate pod1 was picked and pod3 was picked as a prefill node.
	schedulingResult := &fwksched.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"default":                         {TargetEndpoints: []fwksched.Endpoint{endpoint1}},
			experimentalDefaultPrefillProfile: {TargetEndpoints: []fwksched.Endpoint{endpoint3}},
		},
	}
	p.PreRequest(context.Background(), req1, schedulingResult)
	p.wg.Wait()

	// Third request shares partial prefix with first one.
	req3 := &fwksched.InferenceRequest{
		RequestId:   uuid.NewString(),
		TargetModel: "test-model1",
		Body: &fwkrh.InferenceRequestBody{
			Completions: &fwkrh.CompletionsRequest{
				Prompt: fwkrh.Prompt{Raw: "aaaabbbb"},
			},
		},
	}
	_ = p.PrepareRequestData(context.Background(), req3, endpoints)

	// Verify pod1 has the correct prefix match info
	info1, _ := endpoint1.Get(attrprefix.PrefixCacheMatchInfoKey)
	prefixInfo1 := info1.(*attrprefix.PrefixCacheMatchInfo)
	assert.Equal(t, 1, prefixInfo1.MatchBlocks()) // one block ("aaaa") matches
	assert.Equal(t, 2, prefixInfo1.TotalBlocks()) // "aaaabbbb" -> 2 blocks

	// Verify pod3 (prefill node) also has the match
	info3, _ := endpoint3.Get(attrprefix.PrefixCacheMatchInfoKey)
	prefixInfo3 := info3.(*attrprefix.PrefixCacheMatchInfo)
	assert.Equal(t, 1, prefixInfo3.MatchBlocks())

	// Verify pod2 has no match info
	info2, _ := endpoint2.Get(attrprefix.PrefixCacheMatchInfoKey)
	prefixInfo2 := info2.(*attrprefix.PrefixCacheMatchInfo)
	assert.Equal(t, 0, prefixInfo2.MatchBlocks())
}

func TestPrefixPluginChatCompletionsGrowth(t *testing.T) {
	config := config{
		BlockSizeTokens:        2, // Use larger block size
		AutoTune:               false,
		MaxPrefixBlocksToMatch: defaultMaxPrefixBlocks,
		LRUCapacityPerServer:   defaultLRUCapacityPerServer,
	}
	p, _ := newPrepareData(context.Background(), config, nil)

	endpoint1 := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}, &fwkdl.Metrics{}, fwkdl.NewAttributes())
	endpoints := []fwksched.Endpoint{endpoint1}

	// First request with initial conversation
	req1 := &fwksched.InferenceRequest{
		RequestId:   uuid.NewString(),
		TargetModel: "test-model1",
		Body: &fwkrh.InferenceRequestBody{
			ChatCompletions: &fwkrh.ChatCompletionsRequest{
				Messages: []fwkrh.Message{
					{Role: "system", Content: fwkrh.Content{Raw: "You are a helpful assistant"}},
					{Role: "user", Content: fwkrh.Content{Raw: "Hello, how are you?"}},
				},
			},
		},
	}
	_ = p.PrepareRequestData(context.Background(), req1, endpoints)
	state1, _ := plugin.ReadPluginStateKey[*SchedulingContextState](p.PluginState(), req1.RequestId, plugin.StateKey(ApproxPrefixCachePluginType))
	initialHashCount := len(state1.PrefixHashes)
	assert.Greater(t, initialHashCount, 0)

	// Simulate pod1 was picked
	schedulingResult := &fwksched.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"default": {TargetEndpoints: []fwksched.Endpoint{endpoint1}},
		},
	}
	p.PreRequest(context.Background(), req1, schedulingResult)
	p.wg.Wait()

	// Second request adds assistant response and new user message
	req2 := &fwksched.InferenceRequest{
		RequestId:   uuid.NewString(),
		TargetModel: "test-model1",
		Body: &fwkrh.InferenceRequestBody{
			ChatCompletions: &fwkrh.ChatCompletionsRequest{
				Messages: []fwkrh.Message{
					{Role: "system", Content: fwkrh.Content{Raw: "You are a helpful assistant"}},
					{Role: "user", Content: fwkrh.Content{Raw: "Hello, how are you?"}},
					{Role: "assistant", Content: fwkrh.Content{Raw: "I'm doing well, thank you! How can I help you today?"}},
					{Role: "user", Content: fwkrh.Content{Raw: "Can you explain how prefix caching works?"}},
				},
			},
		},
	}
	_ = p.PrepareRequestData(context.Background(), req2, endpoints)
	state2, _ := plugin.ReadPluginStateKey[*SchedulingContextState](p.PluginState(), req2.RequestId, plugin.StateKey(ApproxPrefixCachePluginType))
	extendedHashCount := len(state2.PrefixHashes)
	assert.Greater(t, extendedHashCount, initialHashCount)

	info, _ := endpoint1.Get(attrprefix.PrefixCacheMatchInfoKey)
	prefixInfo := info.(*attrprefix.PrefixCacheMatchInfo)
	assert.Greater(t, prefixInfo.MatchBlocks(), 0, "should have prefix cache hit")
	assert.Equal(t, extendedHashCount, prefixInfo.TotalBlocks())
}

func TestPrefixPluginChatCompletionsMultimodalSameUrlMatches(t *testing.T) {
	config := config{
		BlockSizeTokens:        32,
		AutoTune:               false,
		MaxPrefixBlocksToMatch: defaultMaxPrefixBlocks,
		LRUCapacityPerServer:   defaultLRUCapacityPerServer,
	}
	p, _ := newPrepareData(context.Background(), config, nil)

	endpoint1 := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}, &fwkdl.Metrics{}, fwkdl.NewAttributes())
	endpoints := []fwksched.Endpoint{endpoint1}

	req1 := &fwksched.InferenceRequest{
		RequestId:   uuid.NewString(),
		TargetModel: "test-model1",
		Body: &fwkrh.InferenceRequestBody{
			ChatCompletions: &fwkrh.ChatCompletionsRequest{
				Messages: []fwkrh.Message{
					{
						Role: "user",
						Content: fwkrh.Content{
							Structured: []fwkrh.ContentBlock{
								{Type: "text", Text: "Describe"},
								{Type: "image_url", ImageURL: fwkrh.ImageBlock{Url: "https://storage.googleapis.com/abc1/sample1.jpg"}},
							},
						},
					},
				},
			},
		},
	}
	_ = p.PrepareRequestData(context.Background(), req1, endpoints)
	state1, _ := plugin.ReadPluginStateKey[*SchedulingContextState](p.PluginState(), req1.RequestId, plugin.StateKey(ApproxPrefixCachePluginType))
	initialHashCount := len(state1.PrefixHashes)
	assert.Greater(t, initialHashCount, 0)

	schedulingResult := &fwksched.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"default": {TargetEndpoints: []fwksched.Endpoint{endpoint1}},
		},
	}
	p.PreRequest(context.Background(), req1, schedulingResult)
	p.wg.Wait()

	req2 := &fwksched.InferenceRequest{
		RequestId:   uuid.NewString(),
		TargetModel: "test-model1",
		Body: &fwkrh.InferenceRequestBody{
			ChatCompletions: &fwkrh.ChatCompletionsRequest{
				Messages: []fwkrh.Message{
					{
						Role: "user",
						Content: fwkrh.Content{
							Structured: []fwkrh.ContentBlock{
								{Type: "text", Text: "Describe"},
								{Type: "image_url", ImageURL: fwkrh.ImageBlock{Url: "https://storage.googleapis.com/abc1/sample1.jpg"}},
							},
						},
					},
				},
			},
		},
	}
	_ = p.PrepareRequestData(context.Background(), req2, endpoints)
	info, _ := endpoint1.Get(attrprefix.PrefixCacheMatchInfoKey)
	prefixInfo := info.(*attrprefix.PrefixCacheMatchInfo)

	// Since same prefix hashes are expected to be generated
	assert.Equal(t, prefixInfo.MatchBlocks(), prefixInfo.TotalBlocks())
}

func TestPrefixPluginChatCompletionsMultimodalDifferentUrlPartialMatch(t *testing.T) {
	config := config{
		BlockSizeTokens:        32,
		AutoTune:               false,
		MaxPrefixBlocksToMatch: defaultMaxPrefixBlocks,
		LRUCapacityPerServer:   defaultLRUCapacityPerServer,
	}
	p, _ := newPrepareData(context.Background(), config, nil)

	endpoint1 := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}, &fwkdl.Metrics{}, fwkdl.NewAttributes())
	endpoints := []fwksched.Endpoint{endpoint1}

	req1 := &fwksched.InferenceRequest{
		RequestId:   uuid.NewString(),
		TargetModel: "test-model1",
		Body: &fwkrh.InferenceRequestBody{
			ChatCompletions: &fwkrh.ChatCompletionsRequest{
				Messages: []fwkrh.Message{
					{
						Role: "user",
						Content: fwkrh.Content{
							Structured: []fwkrh.ContentBlock{
								{Type: "text", Text: "Describe"},
								{Type: "image_url", ImageURL: fwkrh.ImageBlock{Url: "https://storage.googleapis.com/bucket1/sample1.jpg"}},
							},
						},
					},
				},
			},
		},
	}
	_ = p.PrepareRequestData(context.Background(), req1, endpoints)
	state1, _ := plugin.ReadPluginStateKey[*SchedulingContextState](p.PluginState(), req1.RequestId, plugin.StateKey(ApproxPrefixCachePluginType))
	initialHashCount := len(state1.PrefixHashes)
	assert.Greater(t, initialHashCount, 0)

	schedulingResult := &fwksched.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"default": {TargetEndpoints: []fwksched.Endpoint{endpoint1}},
		},
	}
	p.PreRequest(context.Background(), req1, schedulingResult)
	p.wg.Wait()

	req2 := &fwksched.InferenceRequest{
		RequestId:   uuid.NewString(),
		TargetModel: "test-model1",
		Body: &fwkrh.InferenceRequestBody{
			ChatCompletions: &fwkrh.ChatCompletionsRequest{
				Messages: []fwkrh.Message{
					{
						Role: "user",
						Content: fwkrh.Content{
							Structured: []fwkrh.ContentBlock{
								{Type: "text", Text: "Describe"},
								{Type: "image_url", ImageURL: fwkrh.ImageBlock{Url: "https://storage.googleapis.com/bucket2/sample2.jpg"}},
							},
						},
					},
					{Role: "assistant", Content: fwkrh.Content{Raw: "This is a sample image."}},
					{Role: "user", Content: fwkrh.Content{Raw: "What else do you see?"}},
				},
			},
		},
	}
	_ = p.PrepareRequestData(context.Background(), req2, endpoints)
	info, _ := endpoint1.Get(attrprefix.PrefixCacheMatchInfoKey)
	prefixInfo := info.(*attrprefix.PrefixCacheMatchInfo)
	// Not a full cache hit as the image url has changed
	assert.Less(t, prefixInfo.MatchBlocks(), prefixInfo.TotalBlocks(), "should not have full prefix cache hit")
}

func TestPrefixPluginAutoTune(t *testing.T) {
	podName := "pod-autotune"
	endpoint := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: podName}},
		&fwkdl.Metrics{
			CacheBlockSize: 16,   // 16 tokens * 4 chars/token = 64 chars per block
			CacheNumBlocks: 1000, // 1000 blocks capacity
		}, fwkdl.NewAttributes())
	endpoints := []fwksched.Endpoint{endpoint}

	req := &fwksched.InferenceRequest{
		RequestId:   uuid.NewString(),
		TargetModel: "test-model",
		Body: &fwkrh.InferenceRequestBody{
			Completions: &fwkrh.CompletionsRequest{
				// Length 128 chars.
				// If block size is 64 chars: 2 blocks
				Prompt: fwkrh.Prompt{Raw: strings.Repeat("a", 128)},
			},
		},
	}

	config := config{
		AutoTune:               true,
		BlockSizeTokens:        32, // Should be ignored in favor of pod metrics (16)
		MaxPrefixBlocksToMatch: defaultMaxPrefixBlocks,
		LRUCapacityPerServer:   1,
	}
	p, _ := newPrepareData(context.Background(), config, nil)

	_ = p.PrepareRequestData(context.Background(), req, endpoints)
	state, _ := plugin.ReadPluginStateKey[*SchedulingContextState](p.PluginState(), req.RequestId, plugin.StateKey(ApproxPrefixCachePluginType))
	// 128 chars / (16 tokens * 4 chars/token) = 2 blocks
	assert.Equal(t, 2, len(state.PrefixHashes), "Should use pod block size (16 tokens) -> 2 body blocks")

	schedulingResult := &fwksched.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"default": {TargetEndpoints: []fwksched.Endpoint{endpoint}},
		},
	}
	p.PreRequest(context.Background(), req, schedulingResult)
	p.wg.Wait()

	// Check indexer state - should be in tracked pods
	assert.Contains(t, p.indexer().Pods(), ServerID(endpoint.GetMetadata().NamespacedName))
}

func TestMaxPrefixTokensToMatch(t *testing.T) {
	// BlockSizeTokens=1 means each block is 4 chars (1 token * 4 chars/token).
	// With MaxPrefixTokensToMatch=2, maxBlocks = 2/1 = 2, so only the first
	// 2 blocks (8 chars) of the prompt should be hashed.
	cfg := config{
		BlockSizeTokens:        1,
		MaxPrefixTokensToMatch: 2,
		LRUCapacityPerServer:   defaultLRUCapacityPerServer,
	}
	p, err := newPrepareData(context.Background(), cfg, nil)
	assert.NoError(t, err)

	endpoint := fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}},
		fwkdl.NewMetrics(), fwkdl.NewAttributes(),
	)

	// Prompt is 16 chars = 4 blocks at blockSize 4 chars, but should be capped to 2.
	req := &fwksched.InferenceRequest{
		RequestId:   uuid.NewString(),
		TargetModel: "test-model",
		Body: &fwkrh.InferenceRequestBody{
			Completions: &fwkrh.CompletionsRequest{
				Prompt: fwkrh.Prompt{Raw: "aaaabbbbccccdddd"},
			},
		},
	}

	err = p.PrepareRequestData(context.Background(), req, []fwksched.Endpoint{endpoint})
	assert.NoError(t, err)

	state, err := plugin.ReadPluginStateKey[*SchedulingContextState](p.PluginState(), req.RequestId, plugin.StateKey(ApproxPrefixCachePluginType))
	assert.NoError(t, err)
	assert.Equal(t, 2, len(state.PrefixHashes), "should cap at MaxPrefixTokensToMatch/BlockSizeTokens = 2 blocks")

	// When MaxPrefixTokensToMatch is 0 (unset), fall back to MaxPrefixBlocksToMatch.
	cfg2 := config{
		BlockSizeTokens:        1,
		MaxPrefixTokensToMatch: 0,
		MaxPrefixBlocksToMatch: 3,
		LRUCapacityPerServer:   defaultLRUCapacityPerServer,
	}
	p2, err := newPrepareData(context.Background(), cfg2, nil)
	assert.NoError(t, err)

	req2 := &fwksched.InferenceRequest{
		RequestId:   uuid.NewString(),
		TargetModel: "test-model",
		Body: &fwkrh.InferenceRequestBody{
			Completions: &fwkrh.CompletionsRequest{
				Prompt: fwkrh.Prompt{Raw: "aaaabbbbccccdddd"},
			},
		},
	}

	err = p2.PrepareRequestData(context.Background(), req2, []fwksched.Endpoint{endpoint})
	assert.NoError(t, err)

	state2, err := plugin.ReadPluginStateKey[*SchedulingContextState](p2.PluginState(), req2.RequestId, plugin.StateKey(ApproxPrefixCachePluginType))
	assert.NoError(t, err)
	assert.Equal(t, 3, len(state2.PrefixHashes), "should fall back to MaxPrefixBlocksToMatch when MaxPrefixTokensToMatch is 0")
}

// BenchmarkPrefixPluginStress is a stress test using prompts of increasing length.
func BenchmarkPrefixPluginStress(b *testing.B) {
	config := config{
		BlockSizeTokens:        1,
		MaxPrefixBlocksToMatch: 50000,
		LRUCapacityPerServer:   defaultLRUCapacityPerServer,
	}
	p, _ := newPrepareData(context.Background(), config, nil)

	promptLen := []int{1024, 4096, 10000, 50000}

	for _, v := range promptLen {
		b.Run(fmt.Sprintf("length_%d", v), func(b *testing.B) {
			prompt := randomPrompt(v)
			endpoint := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{
				NamespacedName: k8stypes.NamespacedName{Name: "pod1"},
			}, nil, fwkdl.NewAttributes())
			endpoints := []fwksched.Endpoint{endpoint}
			req := &fwksched.InferenceRequest{
				RequestId:   uuid.NewString(),
				TargetModel: "model-stress",
				Body: &fwkrh.InferenceRequestBody{
					Completions: &fwkrh.CompletionsRequest{
						Prompt: fwkrh.Prompt{Raw: prompt},
					},
				},
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = p.PrepareRequestData(context.Background(), req, endpoints)
				p.PluginState().Delete(req.RequestId)
			}
		})
	}
}

func randomPrompt(n int) string {
	runes := []rune("abcdefghijklmnopqrstuvwxyz")
	var sb strings.Builder
	for range n {
		sb.WriteRune(runes[rand.Intn(len(runes))])
	}
	return sb.String()
}

func TestPrefixPluginChatCompletionsMultimodalPrefixImageContentMatches(t *testing.T) {
	p360imageContent1 := "data:image/jpeg;base64,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"
	p360imageContent2 := "data:image/jpeg;base64,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"
	config := config{
		BlockSizeTokens:        32,
		AutoTune:               false,
		MaxPrefixBlocksToMatch: defaultMaxPrefixBlocks,
		LRUCapacityPerServer:   defaultLRUCapacityPerServer,
	}
	p, _ := newPrepareData(context.Background(), config, nil)

	endpoint1 := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}, &fwkdl.Metrics{}, fwkdl.NewAttributes())
	endpoints := []fwksched.Endpoint{endpoint1}

	req1 := &fwksched.InferenceRequest{
		RequestId:   uuid.NewString(),
		TargetModel: "test-model1",
		Body: &fwkrh.InferenceRequestBody{
			ChatCompletions: &fwkrh.ChatCompletionsRequest{
				Messages: []fwkrh.Message{
					{
						Role: "user",
						Content: fwkrh.Content{
							Structured: []fwkrh.ContentBlock{
								{Type: "text", Text: "Describe"},
								{Type: "image_url", ImageURL: fwkrh.ImageBlock{Url: p360imageContent1}},
							},
						},
					},
				},
			},
		},
	}
	_ = p.PrepareRequestData(context.Background(), req1, endpoints)
	state1, _ := plugin.ReadPluginStateKey[*SchedulingContextState](p.PluginState(), req1.RequestId, plugin.StateKey(ApproxPrefixCachePluginType))
	initialHashCount := len(state1.PrefixHashes)
	assert.Greater(t, initialHashCount, 0)

	schedulingResult := &fwksched.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"default": {TargetEndpoints: []fwksched.Endpoint{endpoint1}},
		},
	}
	p.PreRequest(context.Background(), req1, schedulingResult)
	p.wg.Wait()

	req2 := &fwksched.InferenceRequest{
		RequestId:   uuid.NewString(),
		TargetModel: "test-model1",
		Body: &fwkrh.InferenceRequestBody{
			ChatCompletions: &fwkrh.ChatCompletionsRequest{
				Messages: []fwkrh.Message{
					{
						Role: "user",
						Content: fwkrh.Content{
							Structured: []fwkrh.ContentBlock{
								{Type: "text", Text: "Describe"},
								{Type: "image_url", ImageURL: fwkrh.ImageBlock{Url: p360imageContent1}},
								{Type: "image_url", ImageURL: fwkrh.ImageBlock{Url: p360imageContent2}},
							},
						},
					},
				},
			},
		},
	}
	_ = p.PrepareRequestData(context.Background(), req2, endpoints)
	info, _ := endpoint1.Get(attrprefix.PrefixCacheMatchInfoKey)
	prefixInfo := info.(*attrprefix.PrefixCacheMatchInfo)

	// Since same prefix hashes are expected to be generated
	assert.Equal(t, 7 , prefixInfo.MatchBlocks())
	assert.Equal(t, 15, prefixInfo.TotalBlocks())

}
