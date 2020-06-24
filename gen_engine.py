from tensorrt.lite import Engine
from tensorrt.infer import LogSeverity
import tensorrt as trt
import uff
from tensorrt.parsers import uffparser


G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.INFO)
uff_model = uff.from_tensorflow_frozen_model("model_save_while_test.pb", ["conv17/output"])

parser = uffparser.create_uff_parser()
parser.register_input("images", (1, 256, 256),0)
parser.register_output("conv17/output")

engine = trt.utils.uff_to_trt_engine(G_LOGGER,
                                     uff_model,
                                     parser,
                                     1,
                                     1<<20,
                                     trt.infer.DataType.FLOAT)

trt.utils.write_engine_to_file("dncnn.engine", engine.serialize())