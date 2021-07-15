import tornado.web
import tornado.ioloop
import test_model

class uploadImgHandler(tornado.web.RequestHandler):
    def post(self):
        files = self.request.files["fileImage"]
        for f in files:
            print("processing ", f.filename)
            fh = open(f"img/{f.filename}", "wb")
            fh.write(f.body)
            fh.close()
        enhanced = test_model.enhance_img(f.filename)
        self.render("success.html", filen = f.filename, result = enhanced)
    def get(self):
        self.render("index.html")
        print("processing index")

if (__name__ == "__main__"):
    app = tornado.web.Application([
        ("/", uploadImgHandler),
        ("/img/(.*)", tornado.web.StaticFileHandler, {'path': 'img'}),
        ("/bootstrap/(.*)", tornado.web.StaticFileHandler, {'path': 'bootstrap'}),
        ("/js/(.*)", tornado.web.StaticFileHandler, {'path': 'js'}),
        ("/css/(.*)", tornado.web.StaticFileHandler, {'path': 'css'}),
        ("/images/(.*)", tornado.web.StaticFileHandler, {'path': 'images'}),
        ("/fonts/(.*)", tornado.web.StaticFileHandler, {'path': 'fonts'}),
        ("/visual_results/(.*)", tornado.web.StaticFileHandler, {'path': 'visual_results'})
    ])

    app.listen(8080)
    print("Listening on port 8080")
    tornado.ioloop.IOLoop.instance().start()