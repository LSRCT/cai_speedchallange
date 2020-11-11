import numpy as np

def calcAVG_generator(generator, verbose=0):
    """
    Calculate average of dataset given by generator
    :param generator: generator with dataset
    :param total_no_of_frames: trivial
    :param verbose: verbose output
    :return: avg of the data
    """
    if verbose:
        print("Calculating average")
    # loop over dataset generator to calculate average
    dataset = generator
    frame_sum = 0
    frame_count = 0
    for batchx, batchy in dataset:
        if len(np.shape(batchx)) > 4:
            for sub_batch in batchx:
                frame_sum += np.sum(sub_batch, axis=0)
        else:
            frame_sum += np.sum(batchx, axis=0)
        frame_count += len(batchx)
    data_avg = (frame_sum / frame_count)
    if verbose:
        print(f"Done with average (shape: {np.shape(data_avg)})")
    return data_avg

def calcAVG_generator_test(generator, verbose=0):
    """
    Calculate average of dataset given by generator
    :param generator: generator with dataset
    :param total_no_of_frames: trivial
    :param verbose: verbose output
    :return: avg of the data
    """
    if verbose:
        print("Calculating average")
    # loop over dataset generator to calculate average
    dataset = generator
    frame_sum = 0
    frame_count = 0
    for batchx in dataset:
        if len(np.shape(batchx)) > 4:
            for sub_batch in batchx:
                frame_sum += np.sum(sub_batch, axis=0)
        else:
            frame_sum += np.sum(batchx, axis=0)
        frame_count += len(batchx)
    data_avg = (frame_sum / frame_count)
    if verbose:
        print(f"Done with average (shape: {np.shape(data_avg)})")
    return data_avg


def calcSTDDEV_generator(generator, avg, verbose=0):
    """
    Calculates standard deviation of a dataset given by a generator.
    Usefull when the dataset doesnt fit into ram
    :param generator: generator with dataset
    :param avg: avg of the dataset
    :param total_no_of_frames: trivial
    :param verbose: verbose output
    :return: standard deviation of dataset
    """
    if verbose:
        print("Calculating standard deviation")
    # loop over datset to calculate standard deviation
    dataset = generator
    data_avg = avg
    diff_from_avg_sum = 0
    frame_count = 0
    for batchx, batchy in dataset:
        if len(np.shape(batchx)) > 4:
            for sub_batch in batchx:
                diff_from_avg_sum += np.sum((sub_batch - data_avg) ** 2, axis=0)
        else:
            diff_from_avg_sum += np.sum((batchx - data_avg) ** 2, axis=0)
        frame_count += len(batchx)
    std_dev = np.sqrt(diff_from_avg_sum / (frame_count - 1))
    if verbose:
        print(f"Done with standard deviation(shape: {np.shape(std_dev)})")
    return std_dev


def calcSTDDEV_generator_test(generator, avg, verbose=0):
    """
    Calculates standard deviation of a dataset given by a generator.
    Usefull when the dataset doesnt fit into ram
    :param generator: generator with dataset
    :param avg: avg of the dataset
    :param total_no_of_frames: trivial
    :param verbose: verbose output
    :return: standard deviation of dataset
    """
    if verbose:
        print("Calculating standard deviation")
    # loop over datset to calculate standard deviation
    dataset = generator
    data_avg = avg
    diff_from_avg_sum = 0
    frame_count = 0
    for batchx in dataset:
        if len(np.shape(batchx)) > 4:
            for sub_batch in batchx:
                diff_from_avg_sum += np.sum((sub_batch - data_avg) ** 2, axis=0)
        else:
            diff_from_avg_sum += np.sum((batchx - data_avg) ** 2, axis=0)
        frame_count += len(batchx)
    std_dev = np.sqrt(diff_from_avg_sum / (frame_count - 1))
    if verbose:
        print(f"Done with standard deviation(shape: {np.shape(std_dev)})")
    return std_dev

def lowmem_generator(pathx, pathy, no_of_frames=1000, mini_batch_size=2, verbose=0, frames_per_file=1000, preprocess_func=0):
    """
    Generator so we dont have to load the whole dataset into memory
    :param pathx: data
    :param pathy: file with labels
    :param no_of_frames: trivial
    :param mini_batch_size: minibatch size for the data batches returned by generator
    :param preprocess_func: function to be applied to data as a preprocessing step. 0 for no preprocessing
    :param verbose: verbose output
    :return: IDK it yields shit
    """
    if verbose:
        print("Gen started")
    ydata_c = []
    with open(pathy, "r") as f:
        for line in f:
            ydata_c.append([float(line.strip("\n"))])
    if verbose:
        print("y done")
    while 1:
        #loop over files
        for i in range(0,int(no_of_frames/frames_per_file)):
            #for i in range(1, 2):
            xdata_dof_raw = np.load(f"{pathx}dof_np//dof_np_{i}.npy", mmap_mode="r")
            xdata_rgb_raw = np.load(f"{pathx}rgb_np//rgb_np_{i}.npy", mmap_mode="r")
            ydata_raw = np.array(ydata_c[i * frames_per_file:(i * frames_per_file) + frames_per_file])
            if verbose:
                print(f"Reading _np_{i}.npy in mmap mode")
            # loop over frames 100 at a time
            for frame in range(0,frames_per_file,100):
                #data_shape  (1000/mini_batch_size, mini_batch_size, 480, 640, 2 oder d)
                shape_param_0 = int(100/mini_batch_size) # cuz 100 frames per file

                #xdata_dof = 1/np.abs(1+np.log10(np.abs(xdata_dof_raw[frame:frame+100].reshape((shape_param_0, mini_batch_size, 480, 640, 2)))+1))
                xdata_dof = np.abs(xdata_dof_raw[frame:frame+100].reshape((shape_param_0, mini_batch_size, 480, 640, 2)))
                if verbose > 1:
                    print("DOF done")

                xdata_rgb = xdata_rgb_raw[frame:frame+100].reshape((shape_param_0, mini_batch_size, 480, 640, 3))#/255
                if verbose > 1:
                    print("RGB done")

                xdata = np.concatenate((xdata_dof, xdata_rgb), axis=-1)
                if not type(preprocess_func) == int:
                    xdata = preprocess_func(xdata)
                ydata = ydata_raw[frame:frame+100].reshape((shape_param_0, mini_batch_size, -1))
                #ydata = one_hot(np.array(ydata_c[i*1000:(i*1000)+1000]), n_classes=32).reshape((shape_param_0, mini_batch_size, -1))
                for xsmall, ysmall in zip(xdata,ydata):
                    yield (xsmall, ysmall)


def preprocess(std_dev, data_avg):
    """
    Implements sklearns standard scaler
    :return: standard scaler function
    """
    def scale_dat(dat):
        return (dat - data_avg) / (std_dev+1E-8)
    return scale_dat


    #np.save(f"rgb_dof_np_{0}.npy", newx)
if __name__ == "__main__":
    pass
