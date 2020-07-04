def compute_color_histograms(cloud, using_hsv=False):

    # Compute histograms for the clusters
    point_colors_list = []

    # Step through each point in the point cloud
    for point in pc2.read_points(cloud, skip_nans=True):
        rgb_list = float_to_rgb(point[3])
        if using_hsv:
            point_colors_list.append(rgb_to_hsv(rgb_list) * 255)
        else:
            point_colors_list.append(rgb_list)

    # Populate lists with color values
    channel_1_vals = []
    channel_2_vals = []
    channel_3_vals = []

    for color in point_colors_list:
        channel_1_vals.append(color[0])
        channel_2_vals.append(color[1])
        channel_3_vals.append(color[2])
    hist1 = np.histogram(channel_1_vals, bins=32, range=(0, 256))
    hist2 = np.histogram(channel_2_vals, bins=32, range=(0, 256))
    hist3 = np.histogram(channel_3_vals, bins=32, range=(0, 256))

    hist_features = np.concatenate((hist1[0], hist2[0], hist3[0])).astype(np.float64)
    norm_features = hist_features / np.sum(hist_features) 
    return normed_features 


def compute_normal_histograms(normal_cloud):
    norm_x_vals = []
    norm_y_vals = []
    norm_z_vals = []

    for norm_component in pc2.read_points(normal_cloud,
                                          field_names = ('normal_x', 'normal_y', 'normal_z'),
                                          skip_nans=True):
        norm_x_vals.append(norm_component[0])
        norm_y_vals.append(norm_component[1])
        norm_z_vals.append(norm_component[2])

    hist1 = np.histogram(norm_x_vals, bins=32, range=(0, 256))
    hist2 = np.histogram(norm_y_vals, bins=32, range=(0, 256))
    hist3 = np.histogram(norm_z_vals, bins=32, range=(0, 256))

    hist_features = np.concatenate((S1_hist[0], S2_hist[0], S3_hist[0])).astype(np.float64)
    normed_features = hist_features / np.sum(hist_features)

    return normed_features