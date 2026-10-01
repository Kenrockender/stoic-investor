# Quote audit

The app's first version had 40 quotes. Many were modern paraphrases that circulate on quote sites, some were duplicates, and a few were never said by the person they were credited to. Each quote was checked against a public-domain translation and then kept, replaced by the translation's real wording, merged into a duplicate, or dropped. The app now has 32 quotes, each with an exact citation.

## Sources

| Author | Translation | Text used |
|---|---|---|
| Marcus Aurelius | George Long (1862) | [Project Gutenberg #15877](https://www.gutenberg.org/ebooks/15877) |
| Epictetus | George Long (1877) | [Project Gutenberg #10661](https://www.gutenberg.org/ebooks/10661) (selection with the full Enchiridion) and the [complete Bell edition on archive.org](https://archive.org/details/discoursesofepic00epicuoft) (OCR text) for the other Discourses and the Fragments |
| Seneca | Richard Mott Gummere (Loeb, 1917) | [Wikisource](https://en.wikisource.org/wiki/Moral_letters_to_Lucilius); section numbers checked against the Latin text at [The Latin Library](https://www.thelatinlibrary.com/sen/seneca.ep1.shtml), because the Wikisource text does not print them |

`python scripts/download_quote_sources.py` downloads these texts into `.quote_sources/` (not committed), and `tests/test_quotes.py` then checks that every quote appears in its source word for word. Line breaks, hyphenation and quote styles are ignored, and "..." marks words left out.

## Decisions

Kept: the wording was already the translation's. Reworded: the quote paraphrased a real passage, so it now uses the translation's exact words. Merged: a duplicate of another entry, whose tags it now shares. Dropped: not found in the author's works.

| Original id | Original text (start) | Decision | Reason | Now |
|---|---|---|---|---|
| ma_001 | You have power over your mind, not outside events... | Dropped | Modern paraphrase with no matching passage in the Meditations | Replaced by ma_101, Meditations 4.7 |
| ma_002 | The impediment to action advances action... | Reworded | Paraphrase of Meditations 5.20 | Meditations 5.20 |
| ma_003 | Very little is needed to make a happy life... | Reworded | Paraphrase of Meditations 7.67 | Meditations 7.67 |
| ma_004 | Never let the future disturb you... | Reworded | Paraphrase of Meditations 7.8 | Meditations 7.8 |
| ma_005 | Loss is nothing else but change... | Reworded | Paraphrase of Meditations 9.35 | Meditations 9.35 |
| ma_006 | Confine yourself to the present. | Reworded | Modernised wording of Meditations 7.29 | Meditations 7.29 |
| ma_007 | If you are distressed by anything external... | Reworded | Paraphrase of Meditations 8.47 | Meditations 8.47 |
| ma_008 | Think of yourself as dead... | Reworded | Paraphrase of Meditations 7.56 | Meditations 7.56 |
| ma_009 | Do not indulge in dreams of what you have not... | Reworded | Paraphrase of Meditations 7.27 | Meditations 7.27 |
| ma_010 | You have within you right now, everything you need... | Dropped | Modern quote with no matching passage | Replaced by ma_102, Meditations 4.49 |
| ma_011 | The object of life is not to be on the side of the majority... | Dropped | A well-known misattribution; not in the Meditations | Replaced by ma_103, Meditations 4.18 |
| ma_012 | Accept the things to which fate binds you... | Reworded | Paraphrase of Meditations 6.39 | Meditations 6.39 |
| ma_013 | Nowhere can man find a quieter or more untroubled retreat... | Reworded | Paraphrase of Meditations 4.3 | Meditations 4.3 |
| sen_001 | It is not that I'm so brave, but that those who make a mistake... | Dropped | Not found in any of the 124 letters | |
| sen_002 | Omnia, Lucili, aliena sunt... Everything, Lucilius, belongs to others... | Reworded | Real (Letter 1.3), but not Gummere's English | Letter 1.3 |
| sen_003 | It is not the man who has too little... | Kept | Gummere's wording | Letter 2.6 |
| sen_004 | Dum differtur vita transcurrit. While we are postponing... | Reworded | Real (Letter 1.2); unverified Latin removed | Letter 1.2 |
| sen_005 | Per aspera ad astra. Through hardship to the stars. | Dropped | Not Seneca; Hercules Furens 437 says "non est ad astra mollis e terris via" | Replaced by ma_102 |
| sen_006 | Nusquam est qui ubique est. One who is everywhere is nowhere. | Reworded | Real (Letter 2.2), but not Gummere's English | Letter 2.2 |
| sen_007 | Recede in te ipse quantum potes. Withdraw into yourself... | Reworded | Real (Letter 7.8), but not Gummere's English | Letter 7.8 |
| sen_008 | Omnia, Lucili, aliena sunt... | Merged | Duplicate of sen_002 | sen_002 |
| sen_009 | Dum differtur vita transcurrit. | Merged | Duplicate of sen_004 | sen_004 |
| sen_010 | Avaritia omnia vitia habet. Greed possesses all vices. | Dropped | Not found in the letters | Replaced by sen_101, Letter 2.6 |
| sen_011 | Vindica te tibi. Claim yourself for yourself. | Reworded | Real (Letter 1.1), but not Gummere's English | Letter 1.1 |
| sen_012 | Recede in te ipse quantum potes; cum his versare... | Merged | Duplicate of sen_007 | sen_007 |
| sen_013 | Nemo ante mortem beatus est. Call no man happy before his death. | Dropped | Solon's saying (Herodotus), not Seneca | Replaced by ma_104, Meditations 4.43 |
| sen_014 | Per aspera ad astra. | Dropped | Duplicate of sen_005, and not Seneca | |
| ep_001 | Seek not that the things which happen should happen as you wish... | Kept | Long's wording | Enchiridion 8 |
| ep_002 | Make the best use of what is in your power... | Reworded | Paraphrase of Discourses 1.1 (credited to the Enchiridion) | Discourses 1.1 |
| ep_003 | He is a wise man who does not grieve for the things which he has not... | Kept | Long's wording | Fragment 129 |
| ep_004 | I laugh at those who think they can damage me... | Reworded | Paraphrase of Discourses 4.5 | Discourses 4.5 |
| ep_005 | Wealth consists not in having great possessions... | Dropped | Not found in the Discourses, Enchiridion or Fragments | |
| ep_006 | First say to yourself what you would be... | Reworded | Paraphrase of Discourses 3.23 | Discourses 3.23 |
| ep_007 | It's not what happens to you, but how you react to it that matters. | Reworded | Modern paraphrase of Enchiridion 5 | Enchiridion 5 |
| ep_008 | No man is free who is not master of himself. | Kept | Long's wording | Fragment 114 |
| ep_009 | Seek not the good in external things; seek it in yourself. | Reworded | Paraphrase of Discourses 3.24 | Discourses 3.24 |
| ep_010 | Demand not that events should happen as you wish... | Merged | Another translation of ep_001 (Enchiridion 8) | ep_001 |
| ep_011 | Preach not to others what they should eat... | Reworded | Paraphrase of Enchiridion 46 | Enchiridion 46 |
| ep_012 | We cannot choose our external circumstances... | Reworded | Modern paraphrase of Enchiridion 1 | Enchiridion 1 |
| ep_013 | Difficulties are things that show a person what they are. | Reworded | Paraphrase of Discourses 1.24 | Discourses 1.24 |

New entries, added so every market mood keeps at least two quotes: ma_101 (Meditations 4.7), ma_102 (4.49), ma_103 (4.18), ma_104 (4.43) and sen_101 (Letter 2.6).

Totals: 4 kept, 23 reworded, 4 merged, 9 dropped, 5 new, so 32 quotes.
