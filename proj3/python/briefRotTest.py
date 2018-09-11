
import scipy as sp
from skimage.transform import rotate

from MatchPics import MatchPics

I1 = sp.ndimage.imread('../data/cv_cover.jpg')

for deg in range(0,91,10):
    I2 = rotate(I1,deg)
    [l1,l2] = MatchPics(I1,I2)