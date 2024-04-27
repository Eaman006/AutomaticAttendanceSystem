import qrcode
t= "https://www.google.co.in/"
img=qrcode.make(t)
type (img)
img.save ("abbc.png")