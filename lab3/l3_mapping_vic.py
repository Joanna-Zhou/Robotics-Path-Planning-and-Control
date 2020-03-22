import os
import glob

import numpy as np
import cv2
import scipy.io as sio
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class FeatureProcessor:
    def __init__(self, data_folder, n_features=500, median_filt_multiplier=1.0):
        # Initiate ORB detector and the brute force matcher
        self.n_features = n_features
        self.median_filt_multiplier = median_filt_multiplier
        self.orb = cv2.ORB_create(nfeatures=n_features)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.data_folder = data_folder
        self.num_images = len(glob.glob(data_folder + '*.jpeg'))

        self.feature_match_locs = []  # [img_i, feat_i, [x, y] of match ((-1, -1) if no match)]

        # store the features found in the first image here. you may find it useful to store both the raw
        # keypoints in kp, and the numpy (u, v) pairs (keypoint.pt) in kp_np
        self.features = dict(kp=[], kp_np=[], des=[])
        self.first_matches = True
        # self.first_image = True
        return

    def get_image(self, id):
        #Load image and convert to grayscale
        img = cv2.imread(os.path.join(self.data_folder,'camera_image_{}.jpeg'.format(id)))
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        return gray

    def get_features(self, id):
        """ Get the keypoints and the descriptors for features for the image with index id."""
        gray = self.get_image(id)

        # cv2.cornerHarris(gray,2,3,0.04)
        # self.harris_kp = cv2.cornerHarris(gray,2,3,0.04)
        # self.img_harris_corner = self.img_next.copy()
        # self.img_harris_corner[dst > 0.01 * dst.max()] = [0, 0, 255]
        # cv2.imshow('Harris Corner Detection', self.img_harris_corner)
        # cv2.waitKey(3)

        kps, descriptors = self.orb.detectAndCompute(gray, None)
        kp_np = [[kp.pt[0],kp.pt[1]] for kp in kps]

        return kps, kp_np, descriptors

    def append_matches(self, matches, new_kp):
        """ Take the current matches and the current keypoints
        and append them to the list of consistent match locations. """
        #img_i, feat_i, [x, y] of match ((-1, -1) if no match)
        # self.feature_match_locs 

        cur_feature_match_locs = np.zeros((len(self.features['kp']), 2))
        cur_feature_match_locs[:] = -1
        for i, match in enumerate(matches):
            cur_feature_match_locs[i] = new_kp[match.queryIdx]
        print("cur_feature_match_locs", cur_feature_match_locs.shape)

        if len(self.feature_match_locs) == 0:
            self.feature_match_locs = np.array([cur_feature_match_locs])
            print("self.feature_match_locs", self.feature_match_locs.shape)
        else:
            self.feature_match_locs = np.append(self.feature_match_locs, np.array([cur_feature_match_locs]), axis = 0)
            print("self.feature_match_locs", self.feature_match_locs.shape)

    def get_matches(self):
        """ Get all of the locations of features matches for each image to the features found in the
        first image. Output should be a numpy array of shape (num_images, num_features_first_image, 2), where
        the actual data is the locations of each feature in each image."""


        for id in range(self.num_images):
            kp, kp_np, des =  self.get_features(id)

            # Match descriptors.
            if self.first_matches:
            #     matches = self.bf.match(self.features['des'], des)

            #     # # Sort them in the order of their distance.
            #     # matches = sorted(matches, key = lambda x:x.distance)

            #     # # Specify number of matches to draw and visualize it
            #     # nmatch2draw = 10
            #     # img_m = cv2.drawMatches(self.img_next, self.kp_next, self.img_cur, self.kp_cur, matches[:10], flags=2)

            #     # cv2.imshow("Feature Matching Window", img_m)
            #     # cv2.waitKey(3)
            #     self.append_matches(matches, kp_np)
            # else:
                self.features['kp'] = kp
                self.features['kp_np'] = kp_np
                self.features['des'] = des
                print("self.features['kp']", len(self.features['kp']))

                # self.feature_match_locs = np.zeros((self.num_images, len(self.features['kp']), 2)) 
                # self.append_matches(matches, kp_np)

                self.first_matches = False

            matches = self.bf.match(self.features['des'], des)
            self.append_matches(matches, kp_np)


        # print("self.feature_match_locs.shape", self.feature_match_locs.shape)
        return self.feature_match_locs


def triangulate(feature_data, tf, inv_K):
    """ For (u, v) image locations of a single feature for every image, as well as the corresponding
    robot poses as T matrices for every image, calculate an estimated xyz position of the feature.

    You're free to use whatever method you like, but we recommend a solution based on least squares, similar
    to section 7.1 of Szeliski's book "Computer Vision: Algorithms and Applications". """

    # raise NotImplementedError('Implement triangulate!')
    # print("tf.shape", tf.shape)    
    # print("tf", tf[0])   
    # print("inv_K", inv_K)  

    factor1 = 0
    factor2 = 0
    for i in range(len(feature_data)):
        camera_center = tf[i][:3,3]
        I = np.identity(3)

        # print("inv_K", inv_K) 
        # print("np.linalg.inv(tf[i])[:3,:]", np.linalg.inv(tf[i])[:3,:].shape)   
        # print("feature_data[i]", feature_data[i]) 
        # print("camera_center", camera_center)

        # I = np.zeros((3,4))
        # I[:3,:3] = np.identity(3)
        # I = np.linalg.inv(I)

        # vs = np.linalg.inv(tf[i]) @ I @ inv_K @ feature_data[i]
        # x = np.ones(3)
        # x[:2] = feature_data[i]
        x = np.append(feature_data[i], 1.)
        inv_R = np.linalg.inv(tf[i][:3, :3])  # see if we need :3 or :
        v = inv_R @ inv_K @ x
        vTv = np.dot(v, v)
        factor1 += I - vTv
        factor2 += np.dot(I - vTv, camera_center)

    p = np.dot(np.linalg.inv(factor1), factor2)

    return p


def main():
    min_feature_views = 20  # minimum number of images a feature must be seen in to be considered useful
    K = np.array([[530.4669406576809, 0.0, 320.5],  # K from sim
                  [0.0, 530.4669406576809, 240.5],
                  [0.0, 0.0, 1.0]])
    inv_K = np.linalg.inv(K)  # will be useful for triangulating feature locations

    # load in data, get consistent feature locations
    data_folder = os.path.join(os.getcwd(),'l3_mapping_data/')
    f_processor = FeatureProcessor(data_folder)
    feature_locations = f_processor.get_matches()  # output shape should be (num_images, num_features, 2)

    # feature rejection
    # raise NotImplementedError('(Optionally) implement feature rejection! (though we strongly recommend it)')
    good_feature_locations = feature_locations
    num_landmarks = len(feature_locations[0])
    print("num_landmarks",num_landmarks )
    print("num_images",f_processor.num_images )
    print("feature_locations",feature_locations.shape )

    pc = np.zeros((num_landmarks, 3))

    # create point cloud map of features
    tf = sio.loadmat("l3_mapping_data/tf.mat")['tf']
    tf_fixed = np.linalg.inv(tf[0, :, :]).dot(tf).transpose((1, 0, 2))

    for i in range(1, num_landmarks):
        # YOUR CODE HERE!! You need to populate good_feature_locations after you reject bad features!
        pc[i] = triangulate(good_feature_locations[:, i, :], tf_fixed, inv_K)

    # ------- PLOTTING TOOLS ------------------------------------------------------------------------------
    # you don't need to modify anything below here unless you change the variable names, or want to modify
    # the plots somehow.

    # get point cloud of trajectory for comparison
    traj_pc = tf_fixed[:, :3, 3]

    # view point clouds with matplotlib
    # set colors based on y positions of point cloud
    max_y = pc[:, 1].max()
    min_y = pc[:, 1].min()
    colors = np.ones((num_landmarks, 4))
    colors[:, :3] = .5
    colors[:, 1] = (pc[:, 1] - min_y) / (max_y - min_y)
    pc_fig = plt.figure()
    ax = pc_fig.add_subplot(111, projection='3d')
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], marker='^', color=colors, label='features')

    ax.scatter(traj_pc[:, 0], traj_pc[:, 1], traj_pc[:, 2], marker='o', label='robot poses')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(elev=-30, azim=-88)
    ax.legend()

    # fit a plane to the feature point cloud for illustration
    # see https://math.stackexchange.com/questions/99299/best-fitting-plane-given-a-set-of-points
    centroid = pc.mean(axis=0)
    pc_minus_cent = pc - centroid # subtract centroid
    u, s, vh = np.linalg.svd(pc_minus_cent.T)
    normal = u.T[2]

    # plot the plane
    # plane is ax + by + cz + d = 0, so z = (-ax - by - d) / c, normal is [a, b, c]
    # normal from svd, point is centroid, get d = -(ax + by + cz)
    d = -centroid.dot(normal)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    X, Y = np.meshgrid(np.linspace(xlim[0], xlim[1], 10),
                       np.linspace(ylim[0], ylim[1], 10))
    Z = (-normal[0] * X - normal[1] * Y - d) * 1. /normal[2]
    ax.plot_wireframe(X,Y,Z, color='k')

    # view all final good features matched on first image (to compare to point cloud)
    feat_fig = plt.figure()
    ax = feat_fig.add_subplot(111)
    ax.imshow(f_processor.get_image(0), cmap='gray')
    ax.scatter(good_feature_locations[0, :, 0], good_feature_locations[0, :, 1], marker='^', color=colors)

    plt.show()

    pc_fig.savefig('point_clouds.png', bbox_inches='tight')
    feat_fig.savefig('feat_fig.png', bbox_inches='tight')

if __name__ == '__main__':
    main()
