from src.preprocessing.preprocessing import build_datasets

train_ds, val_ds, test_ds, class_index = build_datasets()

for images, labels in train_ds.take(1):
    print("Batch image shape:", images.shape)
    print("Batch label shape:", labels.shape)
    print("Pixel value range:", float(images.numpy().min()), "to", float(images.numpy().max()))