from http.server import HTTPServer, BaseHTTPRequestHandler


FILEPATH = '../data/geo_data/dashboard.geojson'
class MapHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        f = open(FILEPATH, "rb").read()
        # Set the referrer policy header
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
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
    run_server()