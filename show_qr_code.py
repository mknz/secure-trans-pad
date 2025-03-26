# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pillow",
#     "qrcode",
# ]
# ///
import argparse
import os
from http.server import SimpleHTTPRequestHandler, HTTPServer
from io import BytesIO

import qrcode

# Directory to store generated QR codes
QR_DIR = "qr_codes"
os.makedirs(QR_DIR, exist_ok=True)


def generate_qr_code(url, name):
    """Generate a QR code for the given URL and save it as an image."""
    img = qrcode.make(url)
    file_path = os.path.join(QR_DIR, f"{name}.png")
    img.save(file_path)
    return file_path


class QRRequestHandler(SimpleHTTPRequestHandler):
    """Custom HTTP handler to serve generated QR codes."""

    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()

            # Generate HTML with QR code images
            html = "<html><body><ul>"
            for file in os.listdir(QR_DIR):
                if file.endswith(".png"):
                    html += f'<h5>{file[:-4]}<br><img src="/{QR_DIR}/{file}" width="200"></h>'
            html += "</body></html>"

            self.wfile.write(html.encode())
        else:
            super().do_GET()


def main():
    parser = argparse.ArgumentParser(
        description="Generate QR codes and serve them over HTTP."
    )
    parser.add_argument(
        "pairs",
        nargs="+",
        metavar=("URL", "NAME"),
        help="URL and name pairs",
        type=str,
    )

    args = parser.parse_args()
    if len(args.pairs) % 2 != 0:
        parser.error("Each URL must have a corresponding name.")

    url_name_pairs = zip(args.pairs[0::2], args.pairs[1::2])

    for url, name in url_name_pairs:
        generate_qr_code(url, name)

    # Start HTTP server
    server_address = ("", 8000)
    httpd = HTTPServer(server_address, QRRequestHandler)
    print("Serving on http://localhost:8000")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
