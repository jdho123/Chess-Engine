from jinja2 import Template
import numpy as np

NUMPY_TO_CPP = {
    # Booleans
    np.bool_: "bool",
    # Fixed-width integers (requires <cstdint> in C++)
    np.int8: "int8_t",
    np.uint8: "uint8_t",
    np.int16: "int16_t",
    np.uint16: "uint16_t",
    np.int32: "int32_t",
    np.uint32: "uint32_t",
    np.int64: "int64_t",
    np.uint64: "uint64_t",
    # Floating point
    np.float32: "float",
    np.float64: "double",
    np.longdouble: "long double",
    # Complex numbers (requires <complex> in C++)
    np.complex64: "std::complex<float>",
    np.complex128: "std::complex<double>",
    np.clongdouble: "std::complex<long double>",
    # Pointers / Size types
    np.intp: "intptr_t",  # Equivalents: std::ptrdiff_t or ssize_t
    np.uintp: "uintptr_t",  # Equivalents: std::size_t or size_t
}


def export_to_header(
    output_path: str, template_path: str, arrays_dict: dict[str, np.ndarray]
):
    layer_data = []

    for name, array in arrays_dict.items():
        raw_bytes = array.tobytes()

        hex_strings = [f"0x{b:02x}" for b in raw_bytes]
        chunks = [
            ", ".join(hex_strings[i : i + 16]) for i in range(0, len(hex_strings), 16)
        ]

        ctype = NUMPY_TO_CPP.get(array.dtype.type)
        if ctype is None:
            raise TypeError(f"Invalid array type: {array.dtype}")

        layer_data.append(
            {"name": name, "size": len(raw_bytes), "chunks": chunks, "c_type": ctype}
        )

    with open(template_path, "r", encoding="utf-8") as f:
        template = Template(f.read())

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(template.render(layers=layer_data))
