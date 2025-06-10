from merlin import QuantumLayer
import torch
import pytest
import perceval as pcvl


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_load_model_on_cuda():
    circuit = pcvl.components.GenericInterferometer(
        4,
        pcvl.components.catalog['mzi phase last'].generate,
        shape=pcvl.InterferometerShape.RECTANGLE
    )

    layer = QuantumLayer(
        input_size=2,
        circuit=circuit,
        output_size=1,
        input_state=[1, 1, 0, 0],
        trainable_parameters=["phi_"],
        device=torch.device("cuda"),
    )
    assert layer.device == torch.device("cuda")
    if len(layer.thetas) > 0:
        assert layer.thetas[0].device == torch.device("cuda", index=0)
    assert layer.computation_process.converter.device == torch.device("cuda")
    assert layer.computation_process.simulation_graph.device == torch.device("cuda")
    assert layer.computation_process.converter.list_rct[0][1].device == torch.device("cuda", index=0)
    assert layer.computation_process.simulation_graph.vectorized_operations[-1][0].device == torch.device("cuda",
                                                                                                          index=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_switch_model_to_cuda():
    circuit = pcvl.components.GenericInterferometer(
                4,
                pcvl.components.catalog['mzi phase last'].generate,
                shape=pcvl.InterferometerShape.RECTANGLE)
    layer = QuantumLayer(
        input_size=2,
        circuit=circuit,
        output_size=1,                                                                                                input_state=[1, 1, 0, 0],
        trainable_parameters=["phi_"],
        device=torch.device("cpu"),
    )
    assert layer.device == torch.device("cpu")
    layer = layer.to(torch.device("cuda"))
    assert layer.device == torch.device("cuda")
    if len(layer.thetas) > 0:
        assert layer.thetas[0].device == torch.device("cuda", index=0)
    assert layer.computation_process.converter.device == torch.device("cuda")
    assert layer.computation_process.simulation_graph.device == torch.device("cuda")
    assert layer.computation_process.converter.list_rct[0][1].device == torch.device("cuda", index=0)
    assert layer.computation_process.simulation_graph.vectorized_operations[-1][0].device == torch.device("cuda", index=0)


