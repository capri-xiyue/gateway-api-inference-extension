{{/*
Common labels
*/}}
{{- define "gateway-api-inference-extension.inferencepool.labels" -}}
app.kubernetes.io/name: {{ include "gateway-api-inference-extension.inferencepool.name" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
{{- end }}

{{/*
Inference extension name
*/}}
{{- define "gateway-api-inference-extension.inferencepool.name" -}}
{{- $base := .Release.Name | default "default-pool" | lower | trim | trunc 40 -}}
{{ $base }}-epp
{{- end -}}


{{/*
Selector labels
*/}}
{{- define "gateway-api-inference-extension.inferencepool.selectorLabels" -}}
inferencepool: {{ include "gateway-api-inference-extension.infernecepool.name" . }}
{{- end -}}
