from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
import torch
from time import time
from cv2 import imwrite
from os.path import basename, splitext
from pathlib import Path
from os import getenv
from PIL import Image, ImageDraw

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=80,
              thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
              device=device)


def get_emb(img):
    # Get cropped and prewhitened image tensor
    img_cropped = mtcnn(img)
    # Calculate embedding (unsqueeze to add batch dimension)
    return resnet(img_cropped.unsqueeze(0))

# create an object for every image in the faces dir.
# embeddings are taken from the image
# and the name of the person is taken from the filename

known_faces = []
folders = Path('faces').glob('*')
for folder in folders:
    name = basename(folder)
    imgs = list(folder.glob('*'))
    print(f'{name}: {len(imgs)}')
    faces = [{'name': name,
              'emb': get_emb(Image.open(f))} for f in imgs]
    known_faces = known_faces + faces

def check_faces(frames):
    """ Return the person which its most likely to be
        and how likely it is to be them.
        returns: (got_face_bool, likelyhood(lower is better), name) """
    # crop out faces from the images
    possible_cropped_faces, probs = mtcnn(frames, return_prob=True)
    print(f'Processed {len(possible_cropped_faces)} images')

    # filter out the images that didnt have a face in them
    cropped_faces = list(filter(lambda f: f['face'] is not None,
                                [{'frame': frames[i],
                                  'face': pf,
                                  'prob': probs[i]} for i, pf in enumerate(possible_cropped_faces)]))
    print(f'Found {len(cropped_faces)} faces')
    if len(cropped_faces) > 0:
        faces = resnet(torch.stack([f['face'] for f in cropped_faces]).to(device))
        face_objects = [{'frame': cp['frame'], 'emb': faces[i]} for i, cp in enumerate(cropped_faces)]
        compare_faces = [compare_face(face) for face in face_objects]
        #for face in compare_faces:
        #   imwrite(f'last_pullup/{face["name"]}-{round(float(face["dist"]),3)}.jpg', face['frame'])
        print('saved faces to last_pullup/')
        return compare_faces
    return []
    

def compare_face(cropped_face):
    # cacluate distance to all known faces
    dists = [torch.dist(cropped_face['emb'], face['emb']) for face in known_faces]
    min_dist = min(dists)
    min_dist_index = np.argmin(dists)
    #imwrite(f'{known_faces[min_dist_index]["name"]}/{round(float(min_dist),3)}.jpg', cropped_face['frame'])
    return {'got_face': True, 'dist': min_dist, 'name': known_faces[min_dist_index]['name'], 'frame': cropped_face['frame']}

