import argparse
import json
import signal
import threading
from pathlib import Path
from .server import DecisionService, http_server

parser = argparse.ArgumentParser(description='Loopback typed decisions using uzu-engine')
parser.add_argument('--config', required=True)
parser.add_argument('--journal', required=True)
parser.add_argument('--port', type=int, default=52280)
args = parser.parse_args()
service = DecisionService(json.loads(Path(args.config).read_text()), args.journal)
server = http_server(service, args.port)
signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=server.shutdown).start())
print(json.dumps({'address': server.server_address, 'inference_engine': 'uzu'}), flush=True)
try:
    server.serve_forever()
except KeyboardInterrupt:
    pass
finally:
    server.server_close()
    service.unload()
