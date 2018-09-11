
def LineEq(lineCoeff,x):
    #a,b,c -> ax + by + c = 0
    a = lineCoeff[0]
    b = lineCoeff[1]
    c = lineCoeff[2]
    #y = (-c-a*x)/b
    y = (-c-b*x)/a    #image is rotated 90 degrees so this is the line eq for that!
    return y #really is returning x
