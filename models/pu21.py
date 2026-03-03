import torch


class PU21Encoder:
    """
    Perceptually Uniform (PU21) encoder/decoder for HDR luminance.
    Pure PyTorch implementation — works on CPU and CUDA tensors.

    Reference: Mantiuk & Azimi, Picture Coding Symposium 2021.
    """

    def __init__(self, encoding_type: str = "banding_glare"):
        if encoding_type == "banding":
            self.par = [1.070275272, 0.4088273932, 0.153224308, 0.2520326168,
                        1.063512885, 1.14115047, 521.4527484]
        elif encoding_type == "banding_glare":
            self.par = [0.353487901, 0.3734658629, 8.277049286e-05, 0.9062562627,
                        0.09150303166, 0.9099517204, 596.3148142]
        elif encoding_type == "peaks":
            self.par = [1.043882782, 0.6459495343, 0.3194584211, 0.374025247,
                        1.114783422, 1.095360363, 384.9217577]
        elif encoding_type == "peaks_glare":
            self.par = [816.885024, 1479.463946, 0.001253215609, 0.9329636822,
                        0.06746643971, 1.573435413, 419.6006374]
        else:
            raise ValueError(f"Unknown encoding type: {encoding_type}")

        self.L_min = 0.005
        self.L_max = 10000.0

    def encode(self, Y: torch.Tensor) -> torch.Tensor:
        """Luminance (nits) -> PU21 encoded value. Element-wise, any shape."""
        Y = Y.clamp(min=self.L_min, max=self.L_max)
        p = self.par
        V = p[6] * (((p[0] + p[1] * Y.pow(p[3])) / (1 + p[2] * Y.pow(p[3]))).pow(p[4]) - p[5])
        return V.clamp(min=0.0)

    def decode(self, V: torch.Tensor) -> torch.Tensor:
        """PU21 encoded value -> luminance (nits). Element-wise, any shape."""
        p = self.par
        V_p = (V / p[6] + p[5]).clamp(min=0.0).pow(1.0 / p[4])
        numerator = (V_p - p[0]).clamp(min=0.0)
        denominator = p[1] - p[2] * V_p
        Y = (numerator / denominator).pow(1.0 / p[3])
        return Y
