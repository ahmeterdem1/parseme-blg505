# BeeParser at MWE-2026 PARSEME 2.0 Subtask 1: Can Cross-Lingual Interactions Improve MWE Identification?

This repository is the official codebase for the paper "BeeParser at MWE-2026 PARSEME 2.0 Subtask 1: Can Cross-Lingual
Interactions Improve MWE Identification?", accepted into [PARSEME 2.0 Shared Task](https://unidive.lisn.upsaclay.fr/doku.php?id=other-events:parseme-st)
proceedings within [EACL 2026 Workshops](https://2026.eacl.org/program/workshops/).

This repository contains the code and resources that we have used to
participate in the PARSEME 2.0 Shared Task on MWE Identification.
Our final implementation on the [leaderboard](https://www.codabench.org/competitions/12003/#) 
can be found under the names *bert-multilingual-trial* and *BeeParser*.

This project is also the term project for the course BLG505 - Natural Language Processing
in Istanbul Technical University.

A skeleton code of what we have run to get our results is given. 

The *parseme* notebook is a single notebook that needs all the code in the correct order to
obtain the results. The *main* notebook however, imports the same but better readable and structured
code from the *utils* package.

## Method Sketch

We have used XLM-RoBERTa-base and finetuned it briefly on the training data of each language.
We have leveraged cross-lingual transfer by training on multiple languages at once. We have observed
that training certain language pairs together, improve MWE identification performance on one language
and degrades it on the other. By applying multilingual training to the BERT and evaluating
on the improved language, and monolingual training to the BERT and evaluating on the degraded language,
we have obtained the results reported on the leaderboard.

The improvements that we have observed, only considering MWE detection and not type classification, are
found to be statistically meaningful under the Welch's t-test.

Other content such as the term project report and term project presentation will be uploaded soon.
