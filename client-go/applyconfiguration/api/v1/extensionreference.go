/*
Copyright The Kubernetes Authors.

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

// Code generated by applyconfiguration-gen. DO NOT EDIT.

package v1

import (
	apiv1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"
)

// ExtensionReferenceApplyConfiguration represents a declarative configuration of the ExtensionReference type for use
// with apply.
type ExtensionReferenceApplyConfiguration struct {
	Group      *apiv1.Group      `json:"group,omitempty"`
	Kind       *apiv1.Kind       `json:"kind,omitempty"`
	Name       *apiv1.ObjectName `json:"name,omitempty"`
	PortNumber *apiv1.PortNumber `json:"portNumber,omitempty"`
}

// ExtensionReferenceApplyConfiguration constructs a declarative configuration of the ExtensionReference type for use with
// apply.
func ExtensionReference() *ExtensionReferenceApplyConfiguration {
	return &ExtensionReferenceApplyConfiguration{}
}

// WithGroup sets the Group field in the declarative configuration to the given value
// and returns the receiver, so that objects can be built by chaining "With" function invocations.
// If called multiple times, the Group field is set to the value of the last call.
func (b *ExtensionReferenceApplyConfiguration) WithGroup(value apiv1.Group) *ExtensionReferenceApplyConfiguration {
	b.Group = &value
	return b
}

// WithKind sets the Kind field in the declarative configuration to the given value
// and returns the receiver, so that objects can be built by chaining "With" function invocations.
// If called multiple times, the Kind field is set to the value of the last call.
func (b *ExtensionReferenceApplyConfiguration) WithKind(value apiv1.Kind) *ExtensionReferenceApplyConfiguration {
	b.Kind = &value
	return b
}

// WithName sets the Name field in the declarative configuration to the given value
// and returns the receiver, so that objects can be built by chaining "With" function invocations.
// If called multiple times, the Name field is set to the value of the last call.
func (b *ExtensionReferenceApplyConfiguration) WithName(value apiv1.ObjectName) *ExtensionReferenceApplyConfiguration {
	b.Name = &value
	return b
}

// WithPortNumber sets the PortNumber field in the declarative configuration to the given value
// and returns the receiver, so that objects can be built by chaining "With" function invocations.
// If called multiple times, the PortNumber field is set to the value of the last call.
func (b *ExtensionReferenceApplyConfiguration) WithPortNumber(value apiv1.PortNumber) *ExtensionReferenceApplyConfiguration {
	b.PortNumber = &value
	return b
}
