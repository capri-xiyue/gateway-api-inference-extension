package approximateprefix

import (
	"fmt"
	"testing"

	"github.com/cespare/xxhash/v2"
	"github.com/stretchr/testify/assert"
	fwkrh "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requesthandling"
	fwksched "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

// width 90, height 180
const imageContent = "data:image/jpeg;base64,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"
func TestGetContentBlocks(t *testing.T) {
	tests := []struct {
		name            string
		request         *fwksched.InferenceRequest
		blockSizeTokens int
		wantBlocks      []ContentBlock
	}{
		{
			name: "Pure Text",
			request: &fwksched.InferenceRequest{
				Body: &fwkrh.InferenceRequestBody{
					ChatCompletions: &fwkrh.ChatCompletionsRequest{
						Messages: []fwkrh.Message{
							{Content: fwkrh.Content{Raw: "aaaabbbbccccdddde"}},
						},
					},
				},
			},
			blockSizeTokens: 2,
			wantBlocks: []ContentBlock{
				{
					TokenIDs:  []string{"aaaa", "bbbb",},
					ExtraHash: "",
					Size:      2,
				},
				{
					TokenIDs:  []string{"cccc", "dddd",},
					ExtraHash: "",
					Size:      2,
				},
				{
					TokenIDs:  []string{"e",},
					ExtraHash: "",
					Size:      2,
				},
			},
		},
		{
			name: "Structured Content - Order Preserved",
			request: &fwksched.InferenceRequest{
				Body: &fwkrh.InferenceRequestBody{
					ChatCompletions: &fwkrh.ChatCompletionsRequest{
						Messages: []fwkrh.Message{
							{
								Content: fwkrh.Content{
									Structured: []fwkrh.ContentBlock{
										{Type: "text", Text: "aaaa"},
										{Type: "image_url", ImageURL: fwkrh.ImageBlock{Url: imageContent}},
										{Type: "text", Text: "bbbbc"},
									},
								},
							},
						},
					},
				},
			},
			blockSizeTokens: 8,
			wantBlocks: []ContentBlock{
				{
					TokenIDs:  []string{"aaaa", "P", "P", "P", "P", "P", "P", "P"},
					ExtraHash: "6b06095a171906fd",
					Size:      8,
				},
				{
					TokenIDs:  []string{"P", "P", "P", "P", "P", "P", "P", "P"},
					ExtraHash: "6b06095a171906fd",
					Size:      8,
				},
				{
					TokenIDs:  []string{"P", "P", "bbbb", "c"},
					ExtraHash: "6b06095a171906fd",
					Size:      8,
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := GetContentBlocks(tt.request, tt.blockSizeTokens)
			assert.Equal(t, tt.wantBlocks, got)
		})
	}
}

func imgHash(url string) string {
	h := xxhash.New()
	_, _ = h.Write([]byte(url))
	return fmt.Sprintf("%x", h.Sum64())
}
