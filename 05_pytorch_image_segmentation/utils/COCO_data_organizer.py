from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class COCO_dataset(Dataset):
    '''
    Characterizes a dataset for PyTorch
    '''
    def __init__(self, list_IDs, data_split, object_classes, coco_API, im_width, im_height, 
                 data_dir, cache_dir, transform = None, augment = None):
        '''
        Initialization
        '''
        self.list_IDs = list_IDs
        self.data_split = data_split
        self.data_dir = data_dir
        self.coco_API = coco_API
        self.cache_dir = cache_dir
        
        self.im_width = im_width
        self.im_height = im_height
        
        # make dictionary of object_classes
        self.objects_dict = dict()
        for i,obj in enumerate(object_classes):
            self.objects_dict.update({obj:i+1})

        # get coco category IDs
        self.COCO_id_dict = dict()
        for obj in object_classes:
            obj_id = coco_train.getCatIds(catNms=obj);
            self.COCO_id_dict.update({obj_id[0]:obj})
        
        # define augmentations
        self.augment = augment
        
        # define transformations
        self.transform = transform
            
    def __len__(self):
        '''
        Denotes the total number of samples
        '''
        return len(self.list_IDs)

    def __getitem__(self, index):
        '''
        Generates one sample of data
        '''
        # Select sample
        ID = self.list_IDs[index]
        ID = int(ID)
        
        # read image from cache and check to see if it exists.
        X = imread(os.path.join(self.cache_dir, self.data_split, str(ID) + '.jpg'), IMREAD_COLOR)
        
        try:
            # if X is an array: the image is in the cache, and we will pass over the except statement
            X.any()

        except AttributeError:
            # If X is not an array: the image is not in the cache.
            # we will load and save the image and mask into the cache            
                
            # Load data and get label
            img = self.coco_API.loadImgs([ID])[0]
            X = imread(os.path.join(self.data_dir, 'images', self.data_split, img['file_name']), IMREAD_COLOR)
            X = cvtColor(X, COLOR_BGR2RGB)

            # generate the GT mask image
            h,w,c = X.shape
            y = np.zeros((h,w))

            # load GT of requested image with only the desired objects labeled
            annIds = self.coco_API.getAnnIds(imgIds=[ID], catIds=self.COCO_id_dict.keys(), iscrowd=False)
            annotations = self.coco_API.loadAnns(annIds)

            # replace COCO label with objects_dict label
            for ann in annotations:
                object_name = self.COCO_id_dict[ann['category_id']]
                objet_label = self.objects_dict[object_name]
                gt = self.coco_API.annToMask(ann)
                y[gt == 1] = objet_label

            imwrite(os.path.join(self.cache_dir, self.data_split, str(ID) + '.jpg'), X)
            imwrite(os.path.join(self.cache_dir, self.data_split, str(ID) + '.png'), y)
            
            # crop images to same size
            X,y = COCO_dataset.Random_Sized_Crop(X, y, self.im_width, self.im_height, static = True)
            
            # add dimenstion to mask as H * W * C
            y = y[...,np.newaxis]

            # apply requested augmentations
            if self.augment:
                random_integer = randint(1,1001)
                X,y = self.augment(X, y, random_integer)

            # convert images to C * H * W
            X = X.transpose((2, 0, 1))
            y = y.transpose((2, 0, 1))
                
            # package into dictionary  
            sample = {'image':X.astype(np.float), 'mask':y.astype(np.float)}

            # apply requested transformations
            if self.transform:
                sample = self.transform(sample)

            return sample
        
        # read images if cache is not empty
        y = imread(os.path.join(self.cache_dir, self.data_split, str(ID) + '.png'))

        y = y[:,:,0]
        
        # crop images to same size
        X,y = COCO_dataset.Random_Sized_Crop(X, y, self.im_width, self.im_height, static = False)

        # add dimenstion to mask as H * W * C
        y = y[...,np.newaxis]

        # apply requested augmentations
        if self.augment:
            random_integer = randint(1,1001)
            X,y = self.augment(X, y, random_integer)

        # convert images to C * H * W
        X = X.transpose((2, 0, 1))
        y = y.transpose((2, 0, 1))
        
        # package into dictionary  
        sample = {'image':X.astype(np.float), 'mask':y.astype(np.float)}

        # apply requested transformations
        if self.transform:
            sample = self.transform(sample)

        return sample
    
    @staticmethod
    def get_classifier_data(object_classes, coco_API):
        '''
        After initialization, the function get_classifier_data is called to generate the dataframe of images that
        contain the classes desired by user
        '''
        # get list of images that contain the desired categories
        data = set()
        for cat in object_classes:
            classId = coco_API.getCatIds(catNms=cat)
            imageIds = coco_API.getImgIds(catIds=classId)
            data.update(imageIds)

        # convert to dataframe and add labels
        data = np.array(list(data), dtype=str)

        return data
    
    @staticmethod
    def find_bad_images(image_list, data_dir, data_split, object_classes, coco_API):
        # find B&W images and Images where mask dims != iamge dims, then remove from training and validation images
        bad_images = []
        for im_Number in image_list:
            # get image number
            im_Number = int(im_Number)

            # read the original image
            img = coco_API.loadImgs(im_Number)[0]
            data_arr = imread(os.path.join(data_dir, 'images', data_split, img['file_name']), IMREAD_COLOR)
            data_arr = cvtColor(data_arr, COLOR_BGR2RGB)

            # read instance annotation
            gt = np.zeros((data_arr.shape[0],data_arr.shape[1]))
            mask = []
            for cat in object_classes:
                catIds = coco_API.getCatIds(catNms=cat);
                annIds = coco_API.getAnnIds(imgIds=[im_Number], catIds=catIds, iscrowd=False)
                anns = coco_API.loadAnns(annIds)
                for ann in anns:
                    mask = coco_API.annToMask(ann)
                    gt[mask == 1] = 1

            # some images have no postivie targets
            if isinstance(mask, list):
#                 print 'bad mask {}'.format(im_Number)
                bad_images.append(str(im_Number))

            # check that the image and mask have the same dimensions
            elif (mask.shape[:2] != data_arr.shape[:2]):
#                 print 'mask shape != image shape {}'.format(im_Number)            
                bad_images.append(str(im_Number))

            # some B&W images have onle 1 channel            
            elif len(data_arr.shape) != 3:
#                 print 'black and white image {}'.format(im_Number)
                bad_images.append(str(im_Number))

            # some B&W images have 3 channels, but they are all equivalent
            elif np.array_equal(data_arr[:,:,0],data_arr[:,:,1]):
#                 print 'black and white image {}'.format(im_Number)
                bad_images.append(str(im_Number))

            else:
                continue

        return bad_images
    
    @staticmethod
    def Random_Sized_Crop(img, mask, width=224, height=224, static = False):
        # define random crop transformation for image augmentation
        # Images in cv2 are defined as columns by rows [width x height]

        if static:
            seed(777)

        assert img.shape[0] == mask.shape[0]
        assert img.shape[1] == mask.shape[1]

        if img.shape[1] < width:
            img = resize(img, (width, img.shape[0]), interpolation = INTER_CUBIC)
            mask = resize(mask, (width, mask.shape[0]), interpolation = INTER_NEAREST)   

        if img.shape[0] < height:
            img = resize(img, (img.shape[1], height), interpolation = INTER_CUBIC)
            mask = resize(mask, (mask.shape[1], height), interpolation = INTER_NEAREST)        

        x = randint(0, img.shape[1] - width)
        y = randint(0, img.shape[0] - height)

        img = img[y:y+height, x:x+width]
        mask = mask[y:y+height, x:x+width]

        return img, mask