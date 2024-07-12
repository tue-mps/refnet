focalLength = [8.620165457832087e+02,1.141913179441716e+03];
principalPoint = [4.852594633049715e+02,3.655790429455455e+02];
imageSize = [540,960];
height = 1.45;
pitch =6;

camIntrinsics = cameraIntrinsics(focalLength,principalPoint,imageSize);
sensor = monoCamera(camIntrinsics,height,'Pitch',pitch);

% distAhead = 30;
% spaceToOneSide = 6;
% bottomOffset = 6;

distAhead = 50;
spaceToOneSide = 22;
bottomOffset = 12;

outView = [bottomOffset,distAhead,-spaceToOneSide,spaceToOneSide];

outImageSize = [NaN,250];

birdsEye = birdsEyeView(sensor,outView,outImageSize);


cam_imagefiles = dir('/home/kach271771/Documents/Matlab_BEV/matlab/dataset/RADIal_sample_Matlab/camera/*.jpg');
nfiles = length(cam_imagefiles);
resultspath = '/home/kach271771/Documents/Matlab_BEV/matlab/dataset/RADIal_sample_Matlab/matlab_output/';

% cam_imagefiles = dir('sampledataset/*.jpg'); 
% nfiles = length(cam_imagefiles);
% resultspath = 'sampledataset/results/resultswithoutdetctions';

for cam_image=1:nfiles
   currentfilename = cam_imagefiles(cam_image).name;
   currentfilename_ = strcat('/home/kach271771/Documents/Matlab_BEV/matlab/dataset/RADIal_sample_Matlab/camera/',currentfilename);
   I = imread(currentfilename_);
   BEV = transformImage(birdsEye,I); % Here the transformation is done
   imwrite(BEV, strcat(resultspath, "/", currentfilename));
end
