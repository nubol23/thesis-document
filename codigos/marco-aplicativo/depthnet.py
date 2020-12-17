img_depth = Image.open(img_depth_path)
img_depth = np.transpose(np.asarray(img_depth, dtype=np.float32), (2, 0, 1))
# Escalando valores entre (0 y 1000)
target = img_depth[0, :, :] + img_depth[1, :, :] * 256 + img_depth[2, :, :] * 256 * 256
# Truncando hasta máximo 30
target = np.clip((target / (256 * 256 * 256 - 1)) * 1000, None, 30)
target = torch.from_numpy(target).float()
