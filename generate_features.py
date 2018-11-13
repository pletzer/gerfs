import argparse
import random
import numpy
from scipy.misc import imsave
from matplotlib import pylab

parser = argparse.ArgumentParser(description='Generate random features.')
parser.add_argument('-l', dest='numLevels', default=1, type=int,
                    help='Number of levels (feature types)')
parser.add_argument('-f', dest='maxNumFeaturesOfType', default=1, type=int,
                    help='Max number of features of a certain type')
parser.add_argument('-x', dest='numXPixels', default=128, type=int,
                    help='Number of x pixels')
parser.add_argument('-y', dest='numYPixels', default=256, type=int,
                    help='Number of y pixels')
parser.add_argument('-s', dest='seed', default=123456789, type=int,
                    help='Random seed')
parser.add_argument('-p', dest='plot', action='store_true',
                    help='Plot data')
parser.add_argument('-v', dest='verbose', action='store_true',
                    help='Run in verbose mode')
parser.add_argument('-o', dest='output', type=str, default='',
                    help='Save image')

args = parser.parse_args()

def toImage(filename, data):
    """
    Convert the array to an image
    @param data array of numpy.int values
    """
    dataCol = numpy.zeros(list(data.shape) + [3,], numpy.int16)
    minVal, maxVal = data.min(), data.max()
    difVal = float(maxVal - minVal)
    if difVal > 0:

        #xx = numpy.array(255*(numpy.array(data, numpy.float64) - minVal)/difVal, numpy.int8)
        xx = (data - minVal) / difVal

        # red
        dataCol[..., 1] = 255*numpy.maximum(0, 2*xx - 1)

        # red
        dataCol[..., 0] = 255*(1 - abs(2*xx - 1))

        # blue
        dataCol[..., 2] = 255*numpy.maximum(0, 1 - 2*xx)

    imsave(filename, dataCol)

class Feature(object):

    def __init__(self):
        
        self.x0 = 0
        self.y0 = 0
        self.a = 0
        self.b = 0
        self.angle = 0.0

    def setCentre(self, x0, y0):
        self.x0, self.y0 = x0, y0

    def setAxisDimensions(self, a, b):
        self.a, self.b = a, b

    def setAngle(self, angle):
        self.angle = angle

    def fill(self, data, val):

        xSize, ySize = data.shape
        xs = numpy.linspace(0., xSize - 1, xSize)
        ys = numpy.linspace(0., ySize - 1, ySize)
        xx, yy = numpy.meshgrid(xs, ys, indexing='ij')

        # displace 
        xx -= self.x0
        yy -= self.y0

        # rotate
        cosa, sina = numpy.cos(self.angle), numpy.sin(self.angle)
        xxp =  xx*cosa + yy*sina
        yyp = -xx*sina + yy*cosa

        # normalise
        xxp /= self.a
        yyp /= self.b

        # one/True inside, 0/False outside
        valid = xxp**2 + yyp**2 < 1.0

        # set the data to val inside the feature
        data = (1 - valid)*data + valid*val
        
        return data



# create the array
data = numpy.zeros((args.numXPixels, args.numYPixels), numpy.int)
random.seed(args.seed)

stats = {}

for level in range(1, args.numLevels + 1):

    # number of features of that type
    n = int(args.maxNumFeaturesOfType * random.random())

    stats[level] = n

    for j in range(n):

        xyMin = min(args.numXPixels, args.numYPixels)

        x0 = int(args.numXPixels * random.random())
        y0 = int(args.numYPixels * random.random())
        a = max(1, int(0.3 * xyMin * random.random()))
        b = max(1, int(0.3 * xyMin * random.random()))
        angle = float(0.5 * numpy.pi * random.random())

        if args.verbose:
            print('level = {} num blobs = {} centre: {}, {} axes: {}, {} angle: {}'.format(level, n, x0, y0, a, b, angle) )

        f = Feature()
        f.setCentre(x0, y0)
        f.setAxisDimensions(a, b)
        f.setAngle(angle)
        data = f.fill(data, level)

for level in stats:
    print('level {} => {} features'.format(level, stats[level]))

if args.output:
    toImage(args.output, data)

if args.plot:
    pylab.pcolor(data.transpose())
    pylab.show()

