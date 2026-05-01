import nbformat

try:
    nb = nbformat.read(r'c:\Users\blaze\Desktop\DeepfakeFinal\VeriLens_Final.ipynb', as_version=4)

    # Find the cell currently containing model_ret
    idx = -1
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'code' and 'model_ret = Sequential' in cell.source:
            idx = i
            break

    orig_markdown_source = "## 9. LRCN Original\n\n`TimeDistributed(Dense(256))` → `Dropout(0.3)` → `LSTM(128)` → `Dropout(0.3)` → `Dense(2, softmax)`\n\nLoss: `categorical_crossentropy`"
    retrained_markdown_source = "## 10. LRCN Retrained (with Focal Loss)"

    orig_code_source = """y_train_cat = to_categorical(y_train, 2)
y_test_cat = to_categorical(y_test, 2)

tf.random.set_seed(SEED)
np.random.seed(SEED)

model_orig = Sequential([
    TimeDistributed(Dense(256, activation='relu'), input_shape=(T, F)),
    Dropout(0.3),
    LSTM(128, return_sequences=False),
    Dropout(0.3),
    Dense(2, activation='softmax', dtype='float32'),
])
model_orig.compile(optimizer=Adam(learning_rate=LRCN_LR),
                   loss='categorical_crossentropy', metrics=['accuracy'])

history_orig = model_orig.fit(X_train, y_train_cat, validation_split=0.2,
                              batch_size=LRCN_BATCH, epochs=LRCN_EPOCHS, verbose=1)

y_proba_orig = model_orig.predict(X_test)
y_pred_orig = np.argmax(y_proba_orig, axis=1)
y_score_orig = y_proba_orig[:, 1]

out_orig = RESULTS_BASE / 'lrcn_orig'
out_orig.mkdir(parents=True, exist_ok=True)
model_orig.save(str(out_orig / 'model.h5'))
metrics_orig = full_eval(y_test, y_pred_orig, y_score_orig,
                         'LRCN-Original', out_orig, history_orig)"""

    if idx != -1:
        # Before idx is the markdown cell "9. LRCN Original"
        nb.cells[idx - 1].source = orig_markdown_source

        # Store the current retrained code
        retrained_code_source = nb.cells[idx].source

        # Replace idx with orig_code
        nb.cells[idx].source = orig_code_source
        nb.cells[idx].outputs = []
        nb.cells[idx].execution_count = None

        # Insert new markdown for retrained at idx+1
        new_md = nbformat.v4.new_markdown_cell(retrained_markdown_source)
        nb.cells.insert(idx + 1, new_md)

        # Insert new code for retrained at idx+2
        new_code = nbformat.v4.new_code_cell(retrained_code_source)
        nb.cells.insert(idx + 2, new_code)

        nbformat.write(nb, r'c:\Users\blaze\Desktop\DeepfakeFinal\VeriLens_Final.ipynb')
        print("Successfully rebuilt the notebook!")
    else:
        print("Couldn't find the target cell.")
except Exception as e:
    print(f"Error: {e}")
