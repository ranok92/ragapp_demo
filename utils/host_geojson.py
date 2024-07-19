from http.server import HTTPServer, BaseHTTPRequestHandler
from argparse import ArgumentParser 
from functools import partial

def argument_parser():
    args = ArgumentParser()
    args.add_argument('--filename', required=True, type=str)
    args.add_argument('--port', default=8000, type=int)
    
    return args.parse_args()


class MapHTTPRequestHandler(BaseHTTPRequestHandler):
    def __init__(self, filepath, *args, **kwargs):
        self.host_file_path=filepath 
        super().__init__(*args, **kwargs)


    def do_GET(self):
        f = open(self.host_file_path, "rb").read()
        # Set the referrer policy header
        self.send_response(200)
        self.send_header('Content-type', 'applictaion/json')
        self.send_header('Referrer-Policy', 'no-referrer')  # Change 'no-referrer' to your desired policy
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(f)
        
    def address_string(self):
        host, port = self.client_address[:2]
        #return socket.getfqdn(host)
        return host
    


def run_server(server_class=HTTPServer, handler_class=MapHTTPRequestHandler, port=8000):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f'Starting server on port {port}...')
    httpd.serve_forever()

if __name__ == '__main__':
    args = argument_parser()
    handler = partial(MapHTTPRequestHandler, f'../data/geo_data/{args.filename}')
    run_server(handler_class=handler, port=args.port)
