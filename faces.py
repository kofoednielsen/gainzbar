from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
import torch
from glob import glob
from os.path import basename, splitext
from PIL import Image, ImageDraw


resnet = InceptionResnetV1(pretrained='vggface2').eval()
mtcnn = MTCNN()


def get_emb(img):
    # Get cropped and prewhitened image tensor
    img_cropped = mtcnn(img)
    # Calculate embedding (unsqueeze to add batch dimension)
    return resnet(img_cropped.unsqueeze(0))

filenames = glob('faces/*')
# create an object for every image in the faces dir.
# embeddings are taken from the image
# and the name of the person is taken from the filename
faces = [{'name': splitext(basename(f))[0],
          'emb': get_emb(Image.open(f))} for f in filenames]

def who_is(frame):
    """ Return the person which its most likely to be
        and how likely it is to be them
        returns: got_face_bool, likelyhood(lower is better), name """
    # try to get embeddings.
    # might fail as it might not
    # be able to detect a face
    try:
        test = get_emb(frame)
    except Exception:
        return (False, 'none', 10)
    dists = [torch.dist(test, face['emb']) for face in faces]
    min_dist = min(dists)
    min_dist_index = np.argmin(dists)
    return (True, min_dist, faces[min_dist_index]['name'])
    

if __name__ == '__main__':
    asbjorn = get_emb('asbjorn.jpg')
    niels = get_emb('niels.png')
    test_imgs = glob('asbjorn/*')
    for img in test_imgs:
        try:
            test = get_emb(img)
            da = torch.dist(test, asbjorn, p=2)
            dn = torch.dist(test, niels, p=2)
            print(f'asbj: {round(float(da), 3)}, niels: {round(float(dn), 3)} for img: {img}')
        except Exception:
            pass


