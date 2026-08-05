# HyTorch Glossary

| PyTorch | HyTorch | Meaning |
|---|---|---|
| `torch` | `hytorch` | Top-level package |
| `torch.tensor(data, ...)` | `hytorch.space(data, ...)` | Construct a runtime value |
| `Tensor` | `Space` | Git-backed statespace value |
| feature width | `SpaceBatch` length | Ordered number of Spaces between layers |
| `dtype` | `mtype` | Numeric representation versus agent model type |
| `device` | `harness` | CPU/CUDA versus agent runtime placement |
| `requires_grad` | `requires_feed` | Whether an activation retains backward provenance |
| `torch.nn` | `hytorch.mn` | Neural-network versus meta-network namespace |
| neuron | agent | One output computation unit |
| `nn.Module` | `mn.Module` | Registered owner with dynamic `forward()` topology |
| `nn.Parameter` | `mn.Parameter` | Registered trainable workspace |
| `model.parameters()` | `model.parameters()` | Iterator passed to an optimizer |
| `nn.Linear(m, n)` | `mn.Linear(m, n)` | Dense mapping from `m` inputs to `n` agents |
| `Linear.weight` | `Linear.weight` | Shape `(out_features,)`; one workspace per agent |
| `Linear.bias` | workspace `AGENTS.md` initializer | Initial mutable direction for each output agent |
| parameter value | workspace directory | Persistent instructions, code, tools, examples, and data |
| `torch.nn.init` | `hytorch.mn.init` | In-place workspace initialization |
| `torch.manual_seed` | `hytorch.manual_seed` | Seed workspace prior initialization |
| autograd tape | retained execution graph | Dynamic forward provenance and saved sessions |
| gradient direction | feedback string | Imperative direction for behavior change |
| `.grad` | `.feed` | Accumulated downstream directions for one workspace |
| loss tensor | `Loss` | Output Space plus terminal directional feedback |
| `loss.backward()` | `loss.backward()` | Update candidates and propagate per-input feedback |
| `optimizer.zero_grad()` | `optimizer.zero_feed()` | Clear feedback and discard an unpromoted candidate |
| `torch.optim.Optimizer` | `hytorch.optim.Optimizer` | Own Parameters and update transaction state |
| `torch.optim.SGD` | `hytorch.optim.DFM` | Gradient descent versus directional feedback mutation |
| `optimizer.step()` | `optimizer.step()` | Promote the completed candidate model branch |
| learning rate `lr` | mutation temperature `temp` | Semantic update scale and sampling temperature |
| optimizer budget | `max_tokens` | Backward agent output-token limit |
| parameter delta | candidate workspace mutation | Proposed change made during backward |
| updated parameter storage | promoted global Git commit | Canonical model generation after `step()` |
| saved forward activations | statespace commit and harness session | Context resumed during backward |
| `state_dict()` | model workspace Git repository | Serialization target; API not yet implemented |

The canonical training form is:

```python
optimizer.zero_feed()
output = model(input)
loss = loss_fn(output, target)
loss.backward()
optimizer.step()
```
