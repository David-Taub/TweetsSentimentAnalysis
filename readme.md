1. I trained a ANN to predict a binary classification of tweets. The ANN was built with embedding + BiLSTM + 1
sized 1d convolution (relu) + GAP (softmax).
2. The preprocessing used some regex manipulation and keras word tokenizer. I chose to disassembly the
hashtags and username using their capital letters when possible, since some hashtags and users might contain
common words that are useful for the context. I trimmed the vocabulary to 10 since there were about 30k words
in the dataset, from which about 10k occurred 3 times or more. I used the 1 sized conv + GAP to allow most
significant word extraction, as asked in part 2 of the challenge. I didn't use pretraining (such as Language
model and word2vec trained weights for embedding) at all but I suspect it will improve the results.
3. main.py attached. With 25% validation set, the validation results are 98.55% accuracy, 80% recall and 89.1%
precision.
4. without the significant words requirement, I could have used deeper networks.

5. list of significant words the algorithm detected:
[('scheme', 3.7238517), ('mrprdox17', 3.8555055), ('blackoasis', 3.8656247), ('cve201711826', 3.8720412),
('orcus', 3.883486), ('skeezer', 3.9264345), ('detected', 3.9347453), ('circulating', 3.9451826), ('apples',
3.9769232), ('scams', 4.0216227), ('plaintext', 4.097296), ('natwest', 4.118416), ('85m', 4.1514897), ('spam',
4.17337), ('email', 4.197956), ('pastebin', 4.2523546), ('demo', 4.2672725), ('robar', 4.344683), ('malware',
4.3609548), ('beware', 4.388825), ('artful', 4.400363), ('sneaky', 4.4283705), ('writeup', 4.520674),
('emails', 4.6481113), ('observe', 4.690175), ('homekit', 4.700489), ('phshng', 4.738463), ('netflix',
4.802521), ('prompts', 4.877919), ('leveraging', 4.8832045), ('url', 4.9720078), ('spamphishing', 4.9793973),
("ddos'd", 4.980723), ('popups', 5.007547), ('cyreninc', 5.211799), ('frankfurt', 5.276775), ('vishing',
5.305642), ('15000', 5.3628335), ('inbox', 5.449429), ('cloudflare', 5.518544), ('backdoor', 5.536265),
('skype', 5.5633407), ('devi', 5.582664), ('dangers', 5.589935), ('rounds', 5.6137733), ('deliver', 5.688657),
('280000', 5.7562995), ('johnpodesta', 5.7736564), ('ios', 5.8009844), ('urgency', 5.8939767), ('hijack',
5.9972224), ('phishing', 6.0282564), ('spoof', 6.0828223), ('freemilk', 6.1956244), ('credentials',
6.4453707), ('indistinguishable', 6.4939146), ('phishingnatwestcom', 6.5957932), ('safari', 6.6155853),
('adverts', 6.620698), ('tangodown', 6.699728), ('csoonline', 6.7015386), ('ja', 6.916121), ('successor',
6.9793468), ('zeroday', 7.0065713), ('netflixthemed', 7.049674), ('databreach', 7.0870247), ('ftp',
7.1778717), ('threatintel', 7.2915206), ('penetrating', 7.47419), ('scam', 7.560195),
('vulnerabilitiessoftware', 7.8569303), ('chats', 7.947272), ('tripwire', 7.956118), ('isbuzz', 7.981169),
('analy', 8.138651), ('mirr', 8.424968), ('badbots', 8.531818), ('hijacks', 8.570978), ("hacker's", 8.625116),
('frenzy', 8.683578), ('myetherwallet', 8.74196), ('jameswtmht', 8.74586), ('shockingly', 8.795076),
('urgently', 9.277191), ('clicks', 9.287351), ('conversations', 9.484925), ('w2s', 9.500202), ('icloud',
9.500557), ('douglasmun', 9.514184), ('iotbased', 9.524768), ('ovhfr', 9.672401), ('craigpollack', 9.71415),
('leveraged', 9.7475395), ('pwn2own', 9.763285), ('tovh', 9.783379), ('monthsand', 9.916921), ('patc',
10.063789), ('deploy', 10.122633), ('itsupportlosangeles', 10.142747), ('plugins', 10.287503)]
