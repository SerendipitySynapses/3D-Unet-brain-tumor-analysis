SEGMENT_CLASSES = {
    0 : 'NOT tumor',
    1 : 'NECROTIC/CORE', # or NON-ENHANCING tumor CORE
    2 : 'EDEMA',
    3 : 'ENHANCING' # original 4 -> converted into 3 later
}

# Custom training loop with progress bar
epochs = 50
steps_per_epoch = len(train_generator)
validation_steps = len(val_generator)

for epoch in range(epochs):
    print(f'Epoch {epoch + 1}/{epochs}')
    with tqdm(total=steps_per_epoch, desc='Training', unit='batch') as pbar:
        for step in range(steps_per_epoch):
            X_batch, y_batch = train_generator[step]
            loss = model.train_on_batch(X_batch, y_batch)
            pbar.set_postfix({'loss': loss[0], 'dice_coefficient': loss[1]})
            pbar.update(1)

    with tqdm(total=validation_steps, desc='Validation', unit='batch') as pbar:
        val_loss = []
        for step in range(validation_steps):
            X_val_batch, y_val_batch = val_generator[step]
            loss = model.test_on_batch(X_val_batch, y_val_batch)
            val_loss.append(loss)
            pbar.set_postfix({'val_loss': loss[0], 'val_dice_coefficient': loss[1]})
            pbar.update(1)

    avg_val_loss = np.mean(val_loss, axis=0)
    print(f'Validation loss: {avg_val_loss[0]}, Validation Dice Coefficient: {avg_val_loss[1]}')

