// numpy dtype name -> TypedArray constructor, shared by both backends.
export const DTYPES = {
  float32: Float32Array,
  float64: Float64Array,
  int8: Int8Array,
  int16: Int16Array,
  int32: Int32Array,
  uint8: Uint8Array,
  uint16: Uint16Array,
  uint32: Uint32Array,
  int64: BigInt64Array,
  uint64: BigUint64Array,
  bool: Uint8Array,
  // No native 16-bit float TypedArray: f16/bf16 cross the API as raw bits.
  float16: Uint16Array,
  bfloat16: Uint16Array,
};
