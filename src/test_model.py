import torch
from prepare import prepare_data
from model import ECGNet
from metrics import accuracy, confusion_matrix, per_class_accuracy
if __name__ == '__main__':
    *_, X_test, y_test = prepare_data()
    model = ECGNet()
    model.load_state_dict(torch.load('results/best_model.pt', map_location='cpu'))
    with torch.no_grad():
        preds = model(X_test).argmax(dim=1)
    cm = confusion_matrix(y_test, preds)
    accs = per_class_accuracy(cm)
    print(f'Test Accuracy: {accuracy(y_test, preds)*100:.2f}%')
    print(f'Balanced Accuracy: {sum(accs)/len(accs)*100:.2f}%')
