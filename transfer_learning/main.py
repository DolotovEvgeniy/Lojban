from __future__ import unicode_literals, print_function, division
from io import open
import random
import argparse

import time
import math

import torch
import torch.nn as nn
from torch import optim

from lang_utils import prepare_data, MAX_LENGTH, SOS_token, EOS_token
from models import EncoderRNN, AttnDecoderRNN, device


def ind2words(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def sent2tensor(lang, sentence):
    indexes = ind2words(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def pair2tensors(pair, langs):
    input_tensor = sent2tensor(langs[0], pair[0])
    target_tensor = sent2tensor(langs[1], pair[1])
    return (input_tensor, target_tensor)


teacher_forcing_ratio = 0.5


def train_batch(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
                max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


def train(encoder, decoder, n_iters, pairs, langs, print_every=1000, plot_every=100, learning_rate=1e-2):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [pair2tensors(random.choice(pairs), langs) for _ in range(n_iters)]
    criterion = nn.NLLLoss()

    lr_drop = False

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train_batch(input_tensor, target_tensor, encoder,
                           decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (time_since(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter > n_iters * 0.75 and not lr_drop:
            learning_rate *= 0.1
            lr_drop = True

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0


def evaluate(encoder, decoder, sentence, langs, max_length=MAX_LENGTH):
    input_lang, output_lang = langs
    with torch.no_grad():
        input_tensor = sent2tensor(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


def evaluate_random(encoder, decoder, pairs, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


def evaluate_all(encoder, decoder, pairs, langs):
    output_sentences = []
    for pair in pairs:
        output_words, attentions = evaluate(encoder, decoder, pair[0], langs)
        output_sentence = ' '.join(output_words)
        output_sentences.append(output_sentence)

    with open("output_translation_val.txt", "w") as f:
        f.write("\n".join(output_sentences))
    return output_sentences

def load_pretrained_model(encoder, decoder, pretrain_encoder, pretrain_decoder):
    encoder_ckpt = torch.load(pretrain_encoder)
    decoder_ckpt = torch.load(pretrain_decoder)

    strict = True
    if encoder.embedding.weight.shape != encoder_ckpt['embedding.weight'].shape:
        print("Number of words in pretrained model vary. Removing embedding layer.")
        del encoder_ckpt['embedding.weight']
        strict = False
    encoder.load_state_dict(encoder_ckpt, strict=strict)

    strict = True
    if decoder.out.weight.shape != decoder_ckpt['out.weight'].shape:
        print("Number of words in pretrained model vary. Removing out layer.")
        del decoder_ckpt['out.weight']
        del decoder_ckpt['out.bias']
        strict = False
    if decoder.embedding.weight.shape != decoder_ckpt['embedding.weight'].shape:
        print("Number of words in pretrained model vary. Removing embedding layer.")
        del decoder_ckpt['embedding.weight']
        strict = False

    decoder.load_state_dict(decoder_ckpt, strict=strict)

hidden_size = 256


def main():
    parser = argparse.ArgumentParser("English - Lojban translation")
    parser.add_argument("--source", default='loj', help="source language data")
    parser.add_argument("--target", default='en', help="target language data")
    parser.add_argument("--iters", type=int, default=100000, help="number of iterations to train")
    parser.add_argument("--no-train", type=bool, default=False, help="Do not perform training. Only validation")
    parser.add_argument("--pretrain-encoder", help="Path to pretrained encoder")
    parser.add_argument("--pretrain-decoder", help="Path to pretrained decoder")
    parser.add_argument("--pretrain-input-words", type=int, help="Number of source language words in pretrained model")
    parser.add_argument("--pretrain-output-words", type=int, help="Number of target language words in pretrained model")
    parser.add_argument("--encoder-ckpt", default="encoder.pth", help="Name of encoder checkpoint filename")
    parser.add_argument("--decoder-ckpt", default="decoder.pth", help="Name of decoder checkpoint filename")
    parser.add_argument("--prefix", default='', help='Prefix, added to data files')
    args = parser.parse_args()

    input_lang, output_lang, pairs, pairs_val = prepare_data(args.source, args.target, prefix=args.prefix)
    langs = (input_lang, output_lang)
    print(random.choice(pairs))

    input_words = args.pretrain_input_words or input_lang.n_words
    output_words = args.pretrain_output_words or output_lang.n_words

    encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)



    if args.pretrain_encoder and args.pretrain_decoder:
        load_pretrained_model(encoder, decoder, args.pretrain_encoder, args.pretrain_decoder)
        
    if not args.no_train:
        train(encoder, decoder, args.iters, pairs, langs, print_every=5000)
        torch.save(encoder.state_dict(), args.encoder_ckpt)
        torch.save(decoder.state_dict(), args.decoder_ckpt)

    evaluate_all(encoder, decoder, pairs_val, langs)


if __name__ == '__main__':
    main()
