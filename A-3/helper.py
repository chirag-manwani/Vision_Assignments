import cv2
import numpy as np
import imutils
import math


def calibrate(
    img,
    pat=(5, 3)
):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, pat, None)
    img2 = img
    corners2 = []
    if ret:
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term)
        img2 = cv2.drawChessboardCorners(img, (7, 6), corners2, ret)
    return img2, corners


def record(
    save_path
):
    cap = cv2.VideoCapture(0)
    frames = []
    while(len(frames) != 300):
        ret, frame = cap.read()
        frames.append(frame)
        print(len(frames))
    h, w, _ = frames[0].shape
    size = (w, h)
    fps = 30
    out = cv2.VideoWriter(save_path,
                          cv2.VideoWriter_fourcc(*'DIVX'),
                          fps,
                          size)
    for frame in frames:
        out.write(frame)
    out.release()


def get_keypoints(
    img
):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()
    kps, des = sift.detectAndCompute(gray, None)

    # kps = np.float32([kp.pt for kp in kps])
    return kps, des


def matchKeypoints(
    kpsA,
    kpsB,
    featuresA,
    featuresB,
    reprojThresh
):
    bf = cv2.BFMatcher(crossCheck=True)
    matches = bf.match(featuresA, featuresB)

    matches.sort(key=lambda x: x.distance, reverse=False)
    kpsA = np.float32([kp.pt for kp in kpsA])
    kpsB = np.float32([kp.pt for kp in kpsB])

    ptsA = np.float32([kpsA[match.queryIdx] for match in matches])
    ptsB = np.float32([kpsB[match.trainIdx] for match in matches])

    (H, status) = cv2.findHomography(ptsA, ptsB,
                                     cv2.RANSAC,
                                     reprojThresh)
    # A = cv2.getAffineTransform(ptsA[:3], ptsB[:3])
    return (matches, H, ptsA, ptsB)


def projection_matrix(camera_parameters, homography):
    """
    From the camera calibration matrix and the estimated homography
    compute the 3D projection matrix
    """
    # Compute rotation along the x and y axis as well as the translation
    homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]
    # normalise vectors
    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l
    # compute the orthonormal basis
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)
    # finally, compute the 3D projection matrix from the model to the current frame
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T
    return np.dot(camera_parameters, projection)


class OBJ:
    def __init__(self, filename, swapyz=False):
        """Loads a Wavefront OBJ file. """
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.faces = []
        material = None
        for line in open(filename, "r"):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'v':
                v = map(float, values[1:4])
                if swapyz:
                    v = v[0], v[2], v[1]
                self.vertices.append(v)
            elif values[0] == 'vn':
                v = map(float, values[1:4])
                if swapyz:
                    v = v[0], v[2], v[1]
                self.normals.append(v)
            elif values[0] == 'vt':
                self.texcoords.append(map(float, values[1:3]))
            #elif values[0] in ('usemtl', 'usemat'):
                #material = values[1]
            #elif values[0] == 'mtllib':
                #self.mtl = MTL(values[1])
            elif values[0] == 'f':
                face = []
                texcoords = []
                norms = []
                for v in values[1:]:
                    w = v.split('/')
                    face.append(int(w[0]))
                    if len(w) >= 2 and len(w[1]) > 0:
                        texcoords.append(int(w[1]))
                    else:
                        texcoords.append(0)
                    if len(w) >= 3 and len(w[2]) > 0:
                        norms.append(int(w[2]))
                    else:
                        norms.append(0)
                #self.faces.append((face, norms, texcoords, material))
                self.faces.append((face, norms, texcoords))


def render(img, obj, projection, model, color=False):
    vertices = obj.vertices
    # print(dir(vertices[0]))
    scale_matrix = np.eye(3) * 3
    h, w = model.shape[:2]

    for face in obj.faces:
        face_vertices = face[0]
        points = [list(vertices[vertex - 1]) for vertex in face_vertices]
        while [] in points:
            points.remove([])
        if(len(points) == 0):
            continue
        points = np.array(points)
        # points = np.array([list(p) for p in points], dtype='float32')
        # print(points)
        points = np.dot(points, scale_matrix)
        rot = np.eye(3)
        rot[2, 2] = 0
        rot[1, 1] = 0
        rot[1, 2] = -1
        rot[2, 1] = 1
        points = np.dot(points, rot)
        # points = scale_matrix @ points
        # render model in the middle of the reference surface. To do so,
        # model points must be displaced
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)
        if color is False:
            cv2.fillConvexPoly(img, imgpts, (137, 27, 211))
        else:
            color = hex_to_rgb(face[-1])
            color = color[::-1] # reverse
            cv2.fillConvexPoly(img, imgpts, color)

    return img


if __name__ == "__main__":
    record('p3.avi')