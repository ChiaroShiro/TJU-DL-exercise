import argparse
import collections
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import rnn as rnn_lstm

start_token = 'G'
end_token = 'E'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POEMS_FILE = os.path.join(BASE_DIR, 'poems.txt')
TANGSHI_FILE = os.path.join(BASE_DIR, 'tangshi.txt')
MODEL_FILE = os.path.join(BASE_DIR, 'poem_generator_rnn')
BEGIN_WORDS = ["日", "红", "山", "夜", "湖", "海", "月"]


def enable_batched_forward():
    def forward(self, sentence, is_test=False):
        if sentence.dim() == 1:
            sentence = sentence.unsqueeze(0)

        batch_input = self.word_embedding_lookup(sentence)
        output, _ = self.rnn_lstm(batch_input)
        out = output.contiguous().view(-1, self.lstm_dim)
        out = F.relu(self.fc(out))
        out = self.softmax(out)

        if is_test:
            return out[-1, :].view(1, -1)
        return out

    rnn_lstm.RNN_model.forward = forward


enable_batched_forward()


def process_poems1(file_name):
    poems = []
    with open(file_name, "r", encoding='utf-8') as f:
        for line in f.readlines():
            try:
                title, content = line.strip().split(':')
                content = content.replace(' ', '')
                if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content or \
                                start_token in content or end_token in content:
                    continue
                if len(content) < 5 or len(content) > 80:
                    continue
                content = start_token + content + end_token
                poems.append(content)
            except ValueError:
                print("error")
                pass

    poems = sorted(poems, key=lambda line: len(line))
    all_words = []
    for poem in poems:
        all_words += [word for word in poem]
    counter = collections.Counter(all_words)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    words, _ = zip(*count_pairs)
    words = words[:len(words)] + (' ',)
    word_int_map = dict(zip(words, range(len(words))))
    poems_vector = [list(map(word_int_map.get, poem)) for poem in poems]
    return poems_vector, word_int_map, words


def process_poems2(file_name):
    poems = []
    with open(file_name, "r", encoding='utf-8') as f:
        for line in f.readlines():
            try:
                line = line.strip()
                if line:
                    content = line.replace(' ', '').replace('，', '').replace('。', '')
                    if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content or \
                                    start_token in content or end_token in content:
                        continue
                    if len(content) < 5 or len(content) > 80:
                        continue
                    content = start_token + content + end_token
                    poems.append(content)
            except ValueError:
                pass

    poems = sorted(poems, key=lambda line: len(line))
    all_words = []
    for poem in poems:
        all_words += [word for word in poem]
    counter = collections.Counter(all_words)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    words, _ = zip(*count_pairs)
    words = words[:len(words)] + (' ',)
    word_int_map = dict(zip(words, range(len(words))))
    poems_vector = [list(map(word_int_map.get, poem)) for poem in poems]
    return poems_vector, word_int_map, words


def resolve_dataset(dataset_name):
    if dataset_name == "tangshi":
        return process_poems2(TANGSHI_FILE)
    return process_poems1(POEMS_FILE)


def iter_batches(poems_vec, batch_size):
    n_chunk = len(poems_vec) // batch_size
    for i in range(n_chunk):
        start_index = i * batch_size
        end_index = start_index + batch_size
        x_data = poems_vec[start_index:end_index]
        y_data = []
        for row in x_data:
            y = row[1:]
            y.append(row[-1])
            y_data.append(y)
        yield x_data, y_data


def get_runtime():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        device_ids = list(range(torch.cuda.device_count()))
    else:
        device = torch.device("cpu")
        device_ids = []
    return device, device_ids


def get_default_batch_size(device_ids):
    if len(device_ids) >= 3:
        return 512
    if len(device_ids) == 2:
        return 256
    if len(device_ids) == 1:
        return 128
    return 64


def make_padded_batch(rows, pad_id, device):
    tensors = [torch.tensor(row, dtype=torch.long) for row in rows]
    batch = nn.utils.rnn.pad_sequence(
        tensors,
        batch_first=True,
        padding_value=pad_id
    )
    if device.type == "cuda":
        return batch.to(device, non_blocking=True)
    return batch.to(device)


def build_model(vocab_size, batch_size, device, device_ids):
    word_embedding = rnn_lstm.word_embedding(
        vocab_length=vocab_size + 1,
        embedding_dim=100
    )
    rnn_model = rnn_lstm.RNN_model(
        batch_sz=batch_size,
        vocab_len=vocab_size + 1,
        word_embedding=word_embedding,
        embedding_dim=100,
        lstm_hidden_dim=128
    )
    rnn_model = rnn_model.to(device)

    if len(device_ids) > 1:
        rnn_model = nn.DataParallel(rnn_model, device_ids=device_ids)
    return rnn_model


def save_model(model, model_path):
    state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(state_dict, model_path)


def run_training(dataset_name="poems", epochs=30, batch_size=None, model_path=MODEL_FILE):
    poems_vector, word_to_int, _ = resolve_dataset(dataset_name)
    device, device_ids = get_runtime()
    batch_size = batch_size or get_default_batch_size(device_ids)
    pad_id = word_to_int[' ']

    torch.manual_seed(5)
    np.random.seed(5)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    print("finish  loadding data")
    print("device:", device)
    print("gpu_count:", len(device_ids))
    print("batch_size:", batch_size)
    print("dataset_size:", len(poems_vector))

    rnn_model = build_model(len(word_to_int), batch_size, device, device_ids)
    optimizer = optim.RMSprop(rnn_model.parameters(), lr=0.01)
    loss_fun = torch.nn.NLLLoss(ignore_index=pad_id)

    for epoch in range(epochs):
        rnn_model.train()
        epoch_loss = 0.0
        n_chunk = len(poems_vector) // batch_size

        for batch, (batch_x, batch_y) in enumerate(iter_batches(poems_vector, batch_size)):
            x = make_padded_batch(batch_x, pad_id, device)
            y = make_padded_batch(batch_y, pad_id, device)

            optimizer.zero_grad(set_to_none=True)
            pre = rnn_model(x)
            loss = loss_fun(pre, y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rnn_model.parameters(), 1.0)
            optimizer.step()

            loss_value = loss.item()
            epoch_loss += loss_value

            if batch % 20 == 0:
                pred_tokens = pre.argmax(dim=1).view(y.size(0), y.size(1))[0].detach().cpu().tolist()
                target_tokens = y[0].detach().cpu().tolist()
                print('prediction', pred_tokens)
                print('b_y       ', target_tokens)
                print('*' * 30)
                save_model(rnn_model, model_path)
                print("epoch", epoch, 'batch number', batch, "loss is:", loss_value)
            else:
                print("epoch", epoch, 'batch number', batch, "loss is:", loss_value)

        avg_loss = epoch_loss / max(n_chunk, 1)
        print("epoch", epoch, "avg loss is:", avg_loss)

    save_model(rnn_model, model_path)
    print("finish  save model")
    return model_path


def to_word(predict, vocabs):
    sample = np.argmax(predict)
    if sample >= len(vocabs):
        sample = len(vocabs) - 1
    return vocabs[sample]


def pretty_print_poem(poem):
    poem = poem.replace(start_token, '').replace(end_token, '')
    poem_sentences = poem.split('。')
    printed = False
    for s in poem_sentences:
        s = s.strip()
        if s:
            printed = True
            if s.endswith('。'):
                print(s)
            else:
                print(s + '。')
    if not printed:
        print(poem)


def gen_poem(begin_word, model_path=MODEL_FILE, dataset_name="poems"):
    poems_vector, word_int_map, vocabularies = resolve_dataset(dataset_name)
    del poems_vector

    if begin_word not in word_int_map:
        raise ValueError(f"开始字 {begin_word} 不在词表中。")

    device, _ = get_runtime()
    rnn_model = build_model(len(word_int_map), 1, device, [])

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"未找到训练好的模型文件 {model_path}，请先执行训练。")

    state_dict = torch.load(model_path, map_location=device)
    rnn_model.load_state_dict(state_dict)
    rnn_model.eval()

    poem = begin_word
    word = begin_word
    with torch.no_grad():
        while word != end_token:
            input_ids = torch.tensor(
                [word_int_map[w] for w in poem],
                dtype=torch.long,
                device=device
            )
            output = rnn_model(input_ids, is_test=True)
            word = to_word(output.squeeze(0).detach().cpu().numpy(), vocabularies)
            poem += word
            if len(poem) > 30:
                break
    return poem


def parse_args():
    parser = argparse.ArgumentParser(description="Tang poem generation with RNN/LSTM")
    parser.add_argument("--mode", choices=["train", "generate", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--dataset", choices=["poems", "tangshi"], default="poems")
    parser.add_argument("--model-path", default=MODEL_FILE)
    parser.add_argument("--begin-words", nargs='*', default=BEGIN_WORDS)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode in ("train", "all"):
        run_training(
            dataset_name=args.dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            model_path=args.model_path
        )

    if args.mode in ("generate", "all"):
        for begin_word in args.begin_words:
            pretty_print_poem(
                gen_poem(
                    begin_word,
                    model_path=args.model_path,
                    dataset_name=args.dataset
                )
            )


if __name__ == "__main__":
    main()
