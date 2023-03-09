import easyocr

img_path = r"C:\Users\91976\Desktop\programming\AI and Ml\projects\ANPR(automatic name plate recognition)\output\output.png"

reader = easyocr.Reader(['en'])
result = reader.readtext(img_path)

print(result[0][1])