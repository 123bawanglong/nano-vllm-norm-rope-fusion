"""Dispatcher contract and CUDA stream/Graph integration for the inference op."""
import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), 'requires CUDA PyTorch')
class DispatcherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.qk_norm_rope import load_extension
        load_extension()

    def inputs(self, device='cuda'):
        return (
            torch.randn(2, 16, 128, device=device, dtype=torch.bfloat16),
            torch.randn(2, 8, 128, device=device, dtype=torch.bfloat16),
            torch.ones(128, device=device, dtype=torch.bfloat16),
            torch.ones(128, device=device, dtype=torch.bfloat16),
            torch.tensor([0, 1], device=device),
            torch.randn(4, 1, 128, device=device),
            1e-6, 4, False,
        )

    def test_schema_and_cuda_dispatch(self):
        op = torch.ops.qk_norm_rope.fused.default
        self.assertEqual(
            str(op._schema),
            'qk_norm_rope::fused(Tensor q, Tensor k, Tensor q_weight, Tensor k_weight, '
            'Tensor positions, Tensor cos_sin_cache, float eps, int warps, '
            'bool native_graph) -> (Tensor, Tensor)',
        )
        self.assertTrue(torch._C._dispatch_has_kernel_for_dispatch_key('qk_norm_rope::fused', 'CUDA'))
        self.assertFalse(torch._C._dispatch_has_kernel_for_dispatch_key('qk_norm_rope::fused', 'CPU'))
        with torch.inference_mode():
            torch.library.opcheck(op, self.inputs(), test_utils=('test_schema',))

    @torch.inference_mode() if torch is not None else lambda f: f
    def test_direct_call_matches_python_api(self):
        from src.qk_norm_rope import fused
        args = self.inputs()
        before = [t.clone() for t in args[:6]]
        direct = torch.ops.qk_norm_rope.fused(*args)
        wrapped = fused(*args[:7], warps=args[7])
        self.assertIsInstance(direct, tuple)
        for actual, expected, source in zip(direct, wrapped, args[:2]):
            self.assertTrue(torch.equal(actual, expected))
            self.assertEqual(actual.shape, source.shape)
            self.assertEqual(actual.dtype, source.dtype)
            self.assertNotEqual(actual.data_ptr(), source.data_ptr())
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(args[:6], before)))

    def test_cpu_dispatch_is_not_supported(self):
        with self.assertRaisesRegex(NotImplementedError, 'CPU'):
            torch.ops.qk_norm_rope.fused(*self.inputs('cpu'))

    def test_cuda_wrapper_rejects_invalid_configuration(self):
        args = self.inputs()
        with self.assertRaisesRegex(RuntimeError, 'warps must be 1 or 4'):
            torch.ops.qk_norm_rope.fused(*args[:7], 2, False)

    @torch.inference_mode() if torch is not None else lambda f: f
    def test_nondefault_stream_and_graph_replay(self):
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            args = self.inputs()
            eager = torch.ops.qk_norm_rope.fused(*args)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                captured = torch.ops.qk_norm_rope.fused(*args)
            graph.replay()
        stream.synchronize()
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(eager, captured)))
        with torch.cuda.stream(stream):
            args[0].neg_()
            args[1].neg_()
            graph.replay()
            changed = torch.ops.qk_norm_rope.fused(*args)
        stream.synchronize()
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(changed, captured)))
        self.assertFalse(torch.equal(eager[0], captured[0]))


if __name__ == '__main__':
    unittest.main()
