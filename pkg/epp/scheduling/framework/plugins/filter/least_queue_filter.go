/*
Copyright 2025 The Kubernetes Authors.

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

package filter

import (
	"context"
	"encoding/json"
	"math"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
)

const (
	LeastQueueFilterType = "least-queue-filter"
)

// compile-time type validation
var _ framework.Filter = &LeastQueueFilter{}

// LeastQueueFilterFactory defines the factory function for LeastQueueFilter.
func LeastQueueFilterFactory(name string, _ json.RawMessage, _ plugins.Handle) (plugins.Plugin, error) {
	return NewLeastQueueFilter().WithName(name), nil
}

// NewLeastQueueFilter initializes a new LeastQueueFilter and returns its pointer.
func NewLeastQueueFilter() *LeastQueueFilter {
	return &LeastQueueFilter{
		typedName: plugins.TypedName{Type: LeastQueueFilterType, Name: LeastQueueFilterType},
	}
}

// LeastQueueFilter finds the max and min queue size of all pods, divides the whole range
// (max-min) by the number of pods, and finds the pods that fall into the first range.
// The intuition is that if there are multiple pods that share similar queue size in the low range,
// we should consider them all instead of the absolute minimum one. This worked better than picking
// the least one as it gives more choices for the next filter, which on aggregate gave better results.
type LeastQueueFilter struct {
	typedName plugins.TypedName
}

// TypedName returns the type and name tuple of this plugin instance.
func (f *LeastQueueFilter) TypedName() plugins.TypedName {
	return f.typedName
}

// WithName sets the name of the filter.
func (f *LeastQueueFilter) WithName(name string) *LeastQueueFilter {
	f.typedName.Name = name
	return f
}

// Filter filters out pods that doesn't meet the filter criteria.
func (f *LeastQueueFilter) Filter(_ context.Context, _ *types.CycleState, _ *types.LLMRequest, pods []types.Pod) []types.Pod {
	filteredPods := []types.Pod{}

	min := math.MaxInt
	max := 0

	for _, pod := range pods {
		if pod.GetMetrics().WaitingQueueSize <= min {
			min = pod.GetMetrics().WaitingQueueSize
		}
		if pod.GetMetrics().WaitingQueueSize >= max {
			max = pod.GetMetrics().WaitingQueueSize
		}
	}

	for _, pod := range pods {
		if pod.GetMetrics().WaitingQueueSize >= min && pod.GetMetrics().WaitingQueueSize <= min+(max-min)/len(pods) {
			filteredPods = append(filteredPods, pod)
		}
	}

	return filteredPods
}
