import os
# mvpath = './Bistro/HR/MV'# 1
# for video in sorted(os.listdir(mvpath)):
#     # Case that object in path is a folder
#     if os.path.isdir(os.path.join(mvpath, video)):
#         frames = sorted(os.listdir(os.path.join(mvpath, video)))
#         for i, old_name in enumerate(frames):
#             # Check if the file is an EXR file
#             if not old_name.endswith('.exr'):
#                 continue
#
#             n = old_name.split('.')[0]
#
#             new_name = f'{int(n):04d}_4_mv.exr'  #2: 0_img , 1_albedo ,2_normal , 3_depth
#
#             old_path = os.path.join(mvpath, video, old_name)
#             new_path = os.path.join(mvpath, video, new_name)
#
#             print(f'Renaming {old_name} to {new_name}')
#             os.rename(old_path, new_path)
#
#             # Reload the file list
#             frames = sorted(os.listdir(os.path.join(mvpath, video)))

#
# depthpath = './Bistro/HR/Depth'# 1
# for video in sorted(os.listdir(depthpath)):
#     # Case that object in path is a folder
#     if os.path.isdir(os.path.join(depthpath, video)):
#         frames = sorted(os.listdir(os.path.join(depthpath, video)))
#         for i, old_name in enumerate(frames):
#             # Check if the file is an EXR file
#             if not old_name.endswith('.exr'):
#                 continue
#
#             n = old_name.split('.')[0]
#             new_name = f'{int(n):04d}_3_depth.exr'
#
#             old_path = os.path.join(depthpath, video, old_name)
#             new_path = os.path.join(depthpath, video, new_name)
#
#             print(f'Renaming {old_name} to {new_name}')
#             os.rename(old_path, new_path)
#
#             # Reload the file list
#             frames = sorted(os.listdir(os.path.join(depthpath, video)))
#
# #

albedopath = './Bistro/HR/Diffuse'# 1
for video in sorted(os.listdir(albedopath)):
    # Case that object in path is a folder
    if os.path.isdir(os.path.join(albedopath, video)):
        frames = sorted(os.listdir(os.path.join(albedopath, video)))
        for i, old_name in enumerate(frames):
            # Check if the file is an EXR file
            if not old_name.endswith('.exr'):
                continue

            n = old_name.split('.')[0]
            new_name = f'{int(n):04d}_1_albedo.exr'

            old_path = os.path.join(albedopath, video, old_name)
            new_path = os.path.join(albedopath, video, new_name)

            print(f'Renaming {old_name} to {new_name}')
            os.rename(old_path, new_path)

            # Reload the file list
            frames = sorted(os.listdir(os.path.join(albedopath, video)))

# normalpath = './Bistro/HR/Normal'# 1
# for video in sorted(os.listdir(normalpath)):
#     # Case that object in path is a folder
#     if os.path.isdir(os.path.join(normalpath, video)):
#         frames = sorted(os.listdir(os.path.join(normalpath, video)))
#         for i, old_name in enumerate(frames):
#             # Check if the file is an EXR file
#             if not old_name.endswith('.exr'):
#                 continue
#
#             n = old_name.split('.')[0]
#             new_name = f'{int(n):04d}_2_normal.exr'
#
#             old_path = os.path.join(normalpath, video, old_name)
#             new_path = os.path.join(normalpath, video, new_name)
#
#             print(f'Renaming {old_name} to {new_name}')
#             os.rename(old_path, new_path)
#
#             # Reload the file list
#             frames = sorted(os.listdir(os.path.join(normalpath, video)))
#
# imgpath = './Bistro/HR/View'  # 1
# for video in sorted(os.listdir(imgpath)):
#     # Case that object in path is a folder
#     if os.path.isdir(os.path.join(imgpath, video)):
#         frames = sorted(os.listdir(os.path.join(imgpath, video)))
#         for i, old_name in enumerate(frames):
#             # Check if the file is an EXR file
#             if not old_name.endswith('.exr'):
#                 continue
#
#             n = old_name.split('.')[0]
#             new_name = f'{int(n):04d}_0_img.exr'
#
#             old_path = os.path.join(imgpath, video, old_name)
#             new_path = os.path.join(imgpath, video, new_name)
#
#             print(f'Renaming {old_name} to {new_name}')
#             os.rename(old_path, new_path)
#
#             # Reload the file list
#             frames = sorted(os.listdir(os.path.join(imgpath, video)))

print('over')