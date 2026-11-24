import os
import shutil

source_dirs = [
    r'D:\Project\Audio_data\archive (2)\for-original\for-original',
    r'D:\Project\Audio_data\archive (2)\for-rerec\for-rerecorded'
]

target_root = r'D:\Project\backend\dataset'

splits = ['training', 'validation', 'testing']
classes = ['real', 'fake']

for split in splits:
    for cls in classes:
        target_path = os.path.join(target_root, split, cls)
        os.makedirs(target_path, exist_ok=True)

        for src_dir in source_dirs:
            src_path = os.path.join(src_dir, split, cls)
            if os.path.exists(src_path):
                for file in os.listdir(src_path):
                    src_file = os.path.join(src_path, file)
                    # Avoid name collision using source folder name
                    src_tag = os.path.basename(os.path.dirname(src_dir))
                    dst_file = os.path.join(target_path, f"{src_tag}_{file}")
                    shutil.copy2(src_file, dst_file)

print("✅ Dataset conversion complete. Files stored in 'D:\\Project\\backend\\dataset'")
