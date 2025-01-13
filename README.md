
![Advanced Decentralized Finance (DeFi) and Artificial Intelligence (AI)_01](https://github.com/user-attachments/assets/f3afb731-e3d2-4302-b9df-a0edc571ea17)
![390194778-eca79677-486b-4184-ba0c-fa6acd099259](https://github.com/user-attachments/assets/bc479937-45c7-4503-8b59-53be8620a710)
**Papers**

- [PIXIU: A Comprehensive Benchmark, Instruction Dataset and Large Language Model for Finance](https://arxiv.org/abs/2306.05443)
- [The FinBen: An Holistic Financial Benchmark for Large Language Models](https://arxiv.org/abs/2402.12659)
- [No Language is an Island: Unifying Chinese and English in Financial Large Language Models, Instruction Data, and Benchmarks](https://arxiv.org/abs/2403.06249)
- [Dólares or Dollars? Unraveling the Bilingual Prowess of Financial LLMs Between Spanish and English](https://arxiv.org/abs/2402.07405)

> Sentiment Analysis

- [FPB (en_fpb)](https://huggingface.co/datasets/TheFinAI/en-fpb)
- [FIQASA (flare_fiqasa)](https://huggingface.co/datasets/TheFinAI/en-fpb)
- [FOMC (flare_fomc)](https://huggingface.co/datasets/TheFinAI/flare-fomc)
- [SemEval-2017 Task5 (flare_tsa)](https://huggingface.co/datasets/TheFinAI/flare-tsa)

> Classification

- [Headlines (flare_headlines)](https://huggingface.co/datasets/TheFinAI/flare-headlines)
- [FinArg ECC Task1 (flare_finarg_ecc_auc)](https://huggingface.co/datasets/TheFinAI/flare-finarg-ecc-auc)
- [FinArg ECC Task2 (flare_finarg_ecc_arc)](https://huggingface.co/datasets/TheFinAI/flare-finarg-ecc-arc)
- [CFA (flare_cfa)](https://huggingface.co/datasets/TheFinAI/flare-cfa)
- [MultiFin EN (flare_multifin_en)](https://huggingface.co/datasets/TheFinAI/flare-multifin-en)
- [M&A (flare_ma)](https://huggingface.co/datasets/TheFinAI/flare-ma)
- [MLESG EN (flare_mlesg)](https://huggingface.co/datasets/TheFinAI/flare-mlesg)

> Knowledge Extraction

- [NER (flare_ner)](https://huggingface.co/datasets/TheFinAI/flare-ner)
- [Finer Ord (flare_finer_ord)](https://huggingface.co/datasets/TheFinAI/flare-finer-ord)
- [FinRED (flare_finred)](https://huggingface.co/datasets/TheFinAI/flare-finred)
- [FinCausal20 Task1 (flare_causal20_sc)](https://huggingface.co/datasets/TheFinAI/flare-causal20-sc)
- [FinCausal20 Task2 (flare_cd)](https://huggingface.co/datasets/TheFinAI/flare-cd)

> Number Understanding

- [FinQA (flare_finqa)](https://huggingface.co/datasets/TheFinAI/flare-finqa)
- [TATQA (flare_tatqa)](https://huggingface.co/datasets/TheFinAI/flare-tatqa)
- [FNXL (flare_fnxl)](https://huggingface.co/datasets/TheFinAI/flare-fnxl)
- [FSRL (flare_fsrl)](https://huggingface.co/datasets/TheFinAI/flare-fsrl)

> Text Summarization

- [ECTSUM (flare_ectsum)](https://huggingface.co/datasets/TheFinAI/flare-ectsum)
- [EDTSUM (flare_edtsum)](https://huggingface.co/datasets/TheFinAI/flare-edtsum)

> Credit Scoring

- [German (flare_german)](https://huggingface.co/datasets/TheFinAI/flare-german)
- [Australian (flare_australian)](https://huggingface.co/datasets/TheFinAI/flare-australian)
- [Lendingclub (flare_cra_lendingclub)](https://huggingface.co/datasets/daishen/cra-lendingclub)
- [Credit Card Fraud (flare_cra_ccf)](https://huggingface.co/datasets/daishen/cra-ccf)
- [ccFraud (flare_cra_ccfraud)](https://huggingface.co/datasets/daishen/cra-ccfraud)
- [Polish (flare_cra_polish)](https://huggingface.co/datasets/daishen/cra-polish)
- [Taiwan Economic Journal (flare_cra_taiwan)](https://huggingface.co/datasets/daishen/cra-taiwan)
- [PortoSeguro (flare_cra_portoseguro)](https://huggingface.co/datasets/daishen/cra-portoseguro)
- [Travle Insurance (flare_cra_travelinsurance)](https://huggingface.co/datasets/daishen/cra-travelinsurance) 

> Forecasting

- [BigData22 for Stock Movement (flare_sm_bigdata)](https://huggingface.co/datasets/TheFinAI/flare-sm-bigdata)
- [ACL18 for Stock Movement (flare_sm_acl)](https://huggingface.co/datasets/TheFinAI/flare-sm-acl)
- [CIKM18 for Stock Movement (flare_sm_cikm)](https://huggingface.co/datasets/TheFinAI/flare-sm-cikm)

## Overview

AI and Decentralized Finance (DeFi) Integration
Overview
This repository demonstrates how to combine Artificial Intelligence (AI) and Decentralized Finance (DeFi) to create an intelligent trading bot on the Solana blockchain. The bot leverages AI-powered predictions to make informed trading decisions, dynamically adjusts position sizes based on volatility, and interacts with DeFi protocols to execute trades securely.

Key features:

AI-Powered Price Predictions: Uses a trained LSTM (Long Short-Term Memory) model to predict future market prices.
Real-Time Market Data: Fetches live data using WebSockets (e.g., Binance) to ensure the system reacts to market changes in real-time.
Risk Management: Implements dynamic position sizing based on volatility, and a trailing stop loss strategy to lock profits.
Smart Contract Interaction: Executes trades using Solana-based smart contracts deployed on the Solana network.
Features
1. Real-Time Data Handling
Fetches live market prices (e.g., BTC/USDT) via WebSocket connections from Binance.
Continuously streams data and updates predictions.
2. AI Integration
Uses a trained LSTM model for market price prediction.
Scales and transforms real-time data before feeding it into the model for predictions.
Predicts next price points and uses this to drive trading decisions.
3. Risk Management
Implements advanced risk management strategies like trailing stop loss and profit-taking.
Position sizes dynamically adjust based on volatility.
Max drawdown control to avoid large losses.
4. Smart Contract Integration
Interacts with Solana smart contracts for executing buy/sell transactions.
Uses Web3.py to build, sign, and send transactions to the Solana network.
Executes trades with specific conditions based on AI predictions and risk management.
Installation
Prerequisites
Python 3.7 or later
Solana wallet and private key (for interacting with DeFi protocols)
Infura or Alchemy account (to connect to the Solana network)
WebSocket-enabled exchange API (e.g., Binance)
DeFi smart contract ABI and address
Required Libraries
You need to install the following libraries:

bash
 
pip install web3 tensorflow numpy pandas scikit-learn websocket-client requests
Configuration
Web3 Setup:

Obtain your Infura/Alchemy API key to connect to the Solana network.
Update the following code in config.py with your Solana details:
python
 
INFURA_PROJECT_ID = "YOUR_INFURA_PROJECT_ID"
CONTRACT_ADDRESS = "0xYourContractAddress"
PRIVATE_KEY = "YourPrivateKey"
Smart Contract ABI:

Save your smart contract ABI in a contract_abi.json file in the project root directory.
AI Model:

Ensure that you have a pre-trained LSTM model. If you don't, you can train the model yourself using historical price data or use an existing model. The trained model should be saved as trained_model.h5.
WebSocket URL:

Replace the WebSocket URL with the URL from your exchange provider (e.g., Binance WebSocket for real-time BTC/USDT prices).
How It Works
1. Real-Time Data Stream:
The system connects to the WebSocket API (e.g., Binance) to get real-time market prices (BTC/USDT, ETH/USDT, etc.) and continuously feeds the price data to the AI model.

python
 
def start_real_time_data_stream():
    url = "wss://stream.binance.com:9443/ws/btcusdt@trade"
    ws = websocket.WebSocketApp(url, on_message=on_message, on_error=on_error, on_close=on_close)
    ws.on_open = on_open
    ws.run_forever()
2. AI Price Prediction:
The real-time price data is transformed and passed to a pre-trained LSTM model for price prediction. The model predicts the next price point for a given asset.

python
 
def process_real_time_data(price):
    current_price = np.array([price]).reshape(1, -1)
    current_price_scaled = scaler.transform(current_price)
    
    # Predict using the LSTM model
    prediction = model.predict(current_price_scaled)
    predicted_price = scaler.inverse_transform(prediction)
3. Risk Management:
The bot dynamically adjusts position size based on volatility and includes a trailing stop loss that moves with the price to lock profits.

python
 
def update_trailing_stop(current_price, entry_price):
    # Adjust trailing stop as price moves up
    if current_price > entry_price:
        stop_loss_price = current_price * (1 - trail_percentage)
        return stop_loss_price
4. Smart Contract Interaction:
The bot interacts with Solana-based smart contracts to execute trades. It signs transactions and sends them to the Solana network using Web3.py.

python
 
def execute_trade(transaction_details):
    transaction = contract.functions.trade(transaction_details['amount'], transaction_details['price'], transaction_details['action']).buildTransaction({
        'from': account,
        'nonce': w3.eth.getTransactionCount(account),
        'gas': 2000000,
        'gasPrice': w3.toWei('10', 'gwei'),
        'chainId': 1  # Mainnet chainId
    })
    signed_txn = w3.eth.account.signTransaction(transaction, private_key)
    tx_hash = w3.eth.sendRawTransaction(signed_txn.rawTransaction)
    return tx_hash
Example Workflow
The bot connects to the WebSocket feed for BTC/USDT prices.
It streams real-time price updates and feeds them to the LSTM model.
The model predicts future price movements.
The bot calculates the risk (e.g., volatility, trailing stop loss).
If conditions are met (e.g., predicted price movement and risk assessment), the bot sends a buy/sell order via an Solana smart contract.
Running the Bot
Once the bot is set up and configured, you can start the trading loop. The bot will continuously fetch data, make predictions, and execute trades according to your defined risk parameters.

python
 
def main_loop():
    start_websocket_thread()
    while True:
        # AI decision-making and trade execution logic here
        pass
Risk Disclosure
This project involves trading cryptocurrencies, which are highly volatile. While AI and risk management strategies can optimize trading decisions, there is no guarantee of profit. Always use caution, and only trade with funds you can afford to lose.

Contributing
Feel free to submit issues, pull requests, or suggest improvements to this project. If you'd like to add more advanced trading strategies, improve the AI model, or integrate more DeFi protocols, contributions are welcome!

### Tasks

| Data                  | Task                             | Raw    | Data Types                | Modalities        | License         | Paper |
| --------------------- | -------------------------------- | ------ | ------------------------- | ----------------- | --------------- | ----- |
| FPB                   | sentiment analysis               | 4,845  | news                      | text              | CC BY-SA 3.0    | [[1]](#1) |
| FiQA-SA               | sentiment analysis               | 1,173  | news headlines, tweets    | text              | Public          | [[2]](#2) |
| TSA | sentiment analysis | 561 | news headlines | text | CC BY-NC-SA 4.0 | [[3]](#3)       |
| FOMC                  | hawkish-dovish classification    | 496    | FOMC transcripts          | text              | CC BY-NC 4.0 | [[4]](#4)       |
| Headlines             | news headline classification     | 11,412 | news headlines            | text              | CC BY-SA 3.0    | [[5]](#5) |
| FinArg-ECC-Task1      | argument unit classification     | 969    | earnings conference call  | text              | CC BY-NC-SA 4.0 | [[6]](#6) |
| FinArg-ECC-Task2      | argument relation classification | 690    | earnings conference call  | text              | CC BY-NC-SA 4.0 | [[6]](#6) |
| Multifin EN        | multi-class classification | 546 | article headlines | text          | Public | [[7]](#7) |
| M&A                     | deal completeness classification  | 500    | news articles, tweets           | text              | Public          | [[8]](#8) |
| MLESG EN                | ESG Issue Identification          | 300    | news articles                   | text              | CC BY-NC-ND     | [[9]](#9) |
| NER                     | named entity recognition          | 1,366  | financial agreements            | text              | CC BY-SA 3.0    | [[10]](#10) |
| Finer Ord             | named entity recognition         | 1,080  | news articles             | text              | CC BY-NC 4.0    | [[11]](#11) |
| FinRED                | relation extraction              | 1,070  | earning call transcipts   | text              | Public          | [[12]](#12) |
| FinCausual 2020 Task1 | causal classification            | 8,630  | news articles, SEC        | text              | CC BY 4.0       | [[13]](#13) |
| FinCausual 2020 Task2 | causal detection                 | 226    | news articles, SEC        | text              | CC BY 4.0       | [[13]](#13) |
| FinQA                 | question answering               | 8,281  | earnings reports          | text, table       | MIT License     | [[14]](#14) |
| TatQA                 | question answering               | 1,670  | financial reports         | text, table       | MIT License     | [[15]](#15) |
| FNXL                  | numeric labeling                 | 318    | SEC                       | text              | Public          | [[16]](#16) |
| FSRL                  | token classification             | 97     | news articles             | text              | MIT License     | [[17]](#17) |
| ECTSUM                | text summarization               | 495    | earning call transcipts   | text              | Public          | [[18]](#18) |
| EDTSUM                | text summarization               | 2000   | news articles             | text              | Public          | [[19]](#19) |
| German                | credit scoring                   | 1000   | credit records            | table             | CC BY 4.0       | [[20]](#20) |
| Australian            | credit scoring                   | 690    | credit records            | table             | CC BY 4.0       | [[21]](#21) |
| Lending Club | credit scoring | 1,3453 | financial information | table | CC0 1.0 | [[22]](#26to32) |
| BigData22             | stock movement prediction        | 7,164  | tweets, historical prices | text, time series | Public          | [[23]](#23) |
| ACL18                 | stock movement prediction        | 27,053 | tweets, historical prices | text, time series | MIT License     | [[24]](#24) |
| CIKM18                | stock movement prediction        | 4,967  | tweets, historical prices | text, time series | Public          | [[25]](#25) |
| ConvFinQA             | multi-turn question answering    | 1,490  | earnings reports          | text, table       | MIT License     | [[26]](#26) |
| Credit Card Fraud     | Fraud Detection                  | 11,392 | financial information     | table             | (DbCL) v1.0     | [[22]](#26to32) |
| ccFraud               | Fraud Detection                  | 10,485 | financial information     | table             | Public          | [[22]](#26to32) |
| Polish                | Financial Distress Identification| 8,681  | financial status features | table             | CC BY 4.0       | [[22]](#26to32) |
|Taiwan Economic Journal| Financial Distress Identification| 6,819  | financial status features | table             | CC BY 4.0       | [[22]](#26to32) |
| PortoSeguro           | Claim Analysis                   | 11,904 | claim and financial information | table             | Public          | [[22]](#26to32) |
| Travel Insurance      | Claim Analysis                   | 12,665 | claim and financial information | table             | (ODbL) v1.0     | [[22]](#26to32) |



<span id="1">1.</span> Pekka Malo, Ankur Sinha, Pekka Korhonen, Jyrki Wallenius, and Pyry Takala. 2014. Good debt or bad debt: Detecting semantic orientations in economic texts. Journal of the Association for Information Science and Technology 65, 4 (2014), 782–796.

<span id="2">2.</span> Macedo Maia, Siegfried Handschuh, André Freitas, Brian Davis, Ross McDermott, Manel Zarrouk, and Alexandra Balahur. 2018. Www’18 open challenge: financial opinion mining and question answering. In Companion proceedings of the the web conference 2018. 1941–1942.

<span id="3">3.</span> Keith Cortis, André Freitas, Tobias Daudert, Manuela Huerlimann, Manel Zarrouk, Siegfried Handschuh, and Brian Davis. 2017. [SemEval-2017 Task 5: Fine-Grained Sentiment Analysis on Financial Microblogs and News](https://aclanthology.org/S17-2089). In *Proceedings of the 11th International Workshop on Semantic Evaluation (SemEval-2017)*, pages 519–535, Vancouver, Canada. Association for Computational Linguistics.

<span id="4">4.</span> Agam Shah, Suvan Paturi, and Sudheer Chava. 2023. [Trillion Dollar Words: A New Financial Dataset, Task & Market Analysis](https://aclanthology.org/2023.acl-long.368). In *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 6664–6679, Toronto, Canada. Association for Computational Linguistics.

<span id="5">5.</span> Ankur Sinha and Tanmay Khandait. 2021. Impact of news on the commodity market: Dataset and results. In Advances in Information and Communication: Proceedings of the 2021 Future of Information and Communication Conference (FICC), Volume 2. Springer, 589–601.

<span id="6">6.</span> Chen C C, Lin C Y, Chiu C J, et al. [Overview of the NTCIR-17 FinArg-1 Task: Fine-grained argument understanding in financial analysis](https://research.nii.ac.jp/ntcir/workshop/OnlineProceedings17/pdf/ntcir/01-NTCIR17-OV-FINARG-ChenC.pdf)[C]//Proceedings of the 17th NTCIR Conference on Evaluation of Information Access Technologies, Tokyo, Japan. 2023.

<span id="7">7.</span> Rasmus Jørgensen, Oliver Brandt, Mareike Hartmann, Xiang Dai, Christian Igel, and Desmond Elliott. 2023. [MultiFin: A Dataset for Multilingual Financial NLP](https://aclanthology.org/2023.findings-eacl.66). In *Findings of the Association for Computational Linguistics: EACL 2023*, pages 894–909, Dubrovnik, Croatia. Association for Computational Linguistics.

<span id="8">8.</span> Yang, L., Kenny, E.M., Ng, T.L., Yang, Y., Smyth, B., & Dong, R. (2020). [Generating Plausible Counterfactual Explanations for Deep Transformers in Financial Text Classification.](https://arxiv.org/abs/2010.12512) *International Conference on Computational Linguistics*.

<span id="9">9.</span> Chung-Chi Chen, Yu-Min Tseng, Juyeon Kang, Anaïs Lhuissier, Min-Yuh Day, Teng-Tsai Tu, and Hsin-Hsi Chen. 2023. Multi-lingual esg issue identification. In *Proceedings of the Fifth Workshop on Financial Tech- nology and Natural Language Processing (FinNLP) and the Second Multimodal AI For Financial Fore- casting (Muffin)*.

<span id="10">10.</span> Julio Cesar Salinas Alvarado, Karin Verspoor, and Timothy Baldwin. 2015. Domain adaption of named entity recognition to support credit risk assessment. In Proceedings of the Australasian Language Technology Association Workshop 2015. 84–90.

<span id="11">11.</span> Shah A, Vithani R, Gullapalli A, et al. Finer: Financial named entity recognition dataset and weak-supervision model[J]. arXiv preprint arXiv:2302.11157, 2023.

<span id="12">12.</span> Sharma, Soumya et al. “FinRED: A Dataset for Relation Extraction in Financial Domain.” *Companion Proceedings of the Web Conference 2022* (2022): n. pag.

<span id="13">13.</span> Dominique Mariko, Hanna Abi-Akl, Estelle Labidurie, Stephane Durfort, Hugues De Mazancourt, and Mahmoud El-Haj. 2020. [The Financial Document Causality Detection Shared Task (FinCausal 2020)](https://aclanthology.org/2020.fnp-1.3). In *Proceedings of the 1st Joint Workshop on Financial Narrative Processing and MultiLing Financial Summarisation*, pages 23–32, Barcelona, Spain (Online). COLING.

<span id="14">14.</span> Zhiyu Chen, Wenhu Chen, Charese Smiley, Sameena Shah, Iana Borova, Dylan Langdon, Reema Moussa, Matt Beane, Ting-Hao Huang, Bryan R Routledge, et al . 2021. FinQA: A Dataset of Numerical Reasoning over Financial Data. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing. 3697–3711.

<span id="15">15.</span> Zhu, Fengbin, Wenqiang Lei, Youcheng Huang, Chao Wang, Shuo Zhang, Jiancheng Lv, Fuli Feng and Tat-Seng Chua. “TAT-QA: A Question Answering Benchmark on a Hybrid of Tabular and Textual Content in Finance.” *ArXiv* abs/2105.07624 (2021): n. pag.

<span id="16">16.</span> Soumya Sharma, Subhendu Khatuya, Manjunath Hegde, Afreen Shaikh, Koustuv Dasgupta, Pawan Goyal, and Niloy Ganguly. 2023. [Financial Numeric Extreme Labelling: A dataset and benchmarking](https://aclanthology.org/2023.findings-acl.219). In *Findings of the Association for Computational Linguistics: ACL 2023*, pages 3550–3561, Toronto, Canada. Association for Computational Linguistics.

<span id="17">17.</span> Matthew Lamm, Arun Chaganty, Christopher D. Manning, Dan Jurafsky, and Percy Liang. 2018. [Textual Analogy Parsing: What’s Shared and What’s Compared among Analogous Facts](https://aclanthology.org/D18-1008). In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing*, pages 82–92, Brussels, Belgium. Association for Computational Linguistics.

<span id="18">18.</span> Rajdeep Mukherjee, Abhinav Bohra, Akash Banerjee, Soumya Sharma, Manjunath Hegde, Afreen Shaikh, Shivani Shrivastava, Koustuv Dasgupta, Niloy Ganguly, Saptarshi Ghosh, and Pawan Goyal. 2022. [ECTSum: A New Benchmark Dataset For Bullet Point Summarization of Long Earnings Call Transcripts](https://aclanthology.org/2022.emnlp-main.748). In *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing*, pages 10893–10906, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

<span id="19">19.</span> Zhihan Zhou, Liqian Ma, and Han Liu. 2021. [Trade the Event: Corporate Events Detection for News-Based Event-Driven Trading](https://aclanthology.org/2021.findings-acl.186). In *Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021*, pages 2114–2124, Online. Association for Computational Linguistics.

<span id="20">20.</span> Hofmann,Hans. (1994). Statlog (German Credit Data). UCI Machine Learning Repository. https://doi.org/10.24432/C5NC77.

<span id="21">21.</span> Quinlan,Ross. Statlog (Australian Credit Approval). UCI Machine Learning Repository. https://doi.org/10.24432/C59012.

<span id="26to32">22.</span> Duanyu Feng, Yongfu Dai, Jimin Huang, Yifang Zhang, Qianqian Xie, Weiguang Han, Alejandro Lopez-Lira, Hao Wang. 2023. Empowering Many, Biasing a Few: Generalist Credit Scoring through Large Language Models. *ArXiv* abs/2310.00566 (2023): n. pag.

<span id="23">23.</span> Yejun Soun, Jaemin Yoo, Minyong Cho, Jihyeong Jeon, and U Kang. 2022. Accurate Stock Movement Prediction with Self-supervised Learning from Sparse Noisy Tweets. In 2022 IEEE International Conference on Big Data (Big Data). IEEE, 1691–1700.

<span id="24">24.</span> Yumo Xu and Shay B Cohen. 2018. Stock movement prediction from tweets and historical prices. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 1970–1979.

<span id="25">25.</span> Huizhe Wu, Wei Zhang, Weiwei Shen, and Jun Wang. 2018. Hybrid deep sequential modeling for social text-driven stock prediction. In Proceedings of the 27th ACM international conference on information and knowledge management. 1627–1630.

<span id="26">26.</span> Zhiyu Chen, Shiyang Li, Charese Smiley, Zhiqiang Ma, Sameena Shah, and William Yang Wang. 2022. ConvFinQA: Exploring the Chain of Numerical Reasoning in Conversational Finance Question Answering. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 6279–6292, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.




### Evaluation

#### Preparation

##### Locally install
```bash
git clone https://github.com/The-FinAI/PIXIU.git --recursive
cd PIXIU
pip install -r requirements.txt
cd src/financial-evaluation
pip install -e .[multilingual]
```
##### Docker image
```bash
sudo bash scripts/docker_run.sh
```
Above command starts a docker container, you can modify `docker_run.sh` to fit your environment. We provide pre-built image by running `sudo docker pull tothemoon/pixiu:latest`

```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    --network host \
    --env https_proxy=$https_proxy \
    --env http_proxy=$http_proxy \
    --env all_proxy=$all_proxy \
    --env HF_HOME=$hf_home \
    -it [--rm] \
    --name pixiu \
    -v $pixiu_path:$pixiu_path \
    -v $hf_home:$hf_home \
    -v $ssh_pub_key:/root/.ssh/authorized_keys \
    -w $workdir \
    $docker_user/pixiu:$tag \
    [--sshd_port 2201 --cmd "echo 'Hello, world!' && /bin/bash"]
```
Arguments explain:
- `[]` means ignoreable arguments
- `HF_HOME`: huggingface cache dir
- `sshd_port`: sshd port of the container, you can run `ssh -i private_key -p $sshd_port root@$ip` to connect to the container, default to 22001
- `--rm`: remove the container when exit container (ie.`CTRL + D`)

#### Automated Task Assessment
Before evaluation, please download [BART checkpoint](https://drive.google.com/u/0/uc?id=1_7JfF7KOInb7ZrxKHIigTMR4ChVET01m&export=download) to `src/metrics/BARTScore/bart_score.pth`.

For automated evaluation, please follow these instructions:

1. Huggingface Transformer

   To evaluate a model hosted on the HuggingFace Hub (for instance, finma-7b-full), use this command:

```bash
python eval.py \
    --model "hf-causal-llama" \
    --model_args "use_accelerate=True,pretrained=TheFinAI/finma-7b-full,tokenizer=TheFinAI/finma-7b-full,use_fast=False" \
    --tasks "flare_ner,flare_sm_acl,flare_fpb"
```

More details can be found in the [lm_eval](https://github.com/EleutherAI/lm-evaluation-harness) documentation.

2. Commercial APIs


Please note, for tasks such as NER, the automated evaluation is based on a specific pattern. This might fail to extract relevant information in zero-shot settings, resulting in relatively lower performance compared to previous human-annotated results.

```bash
export OPENAI_API_SECRET_KEY=YOUR_KEY_HERE
python eval.py \
    --model gpt-4 \
    --tasks flare_ner,flare_sm_acl,flare_fpb
```

3. Self-Hosted Evaluation

To run inference backend:

```bash
bash scripts/run_interface.sh
```

Please adjust run_interface.sh according to your environment requirements.

To evaluate:

```bash
python data/*/evaluate.py
```

### Create new tasks

Creating a new task for FinBen involves creating a Huggingface dataset and implementing the task in a Python file. This guide walks you through each step of setting up a new task using the FinBen framework.

#### Creating your dataset in Huggingface

Your dataset should be created in the following format:

```python
{
    "query": "...",
    "answer": "...",
    "text": "..."
}
```

In this format:

- `query`: Combination of your prompt and text
- `answer`: Your label

For **Multi-turn** tasks (such as )

For **Classification** tasks (such as [FPB (FinBen_fpb)](https://huggingface.co/datasets/TheFinAI/flare-fpb)), additional keys should be defined:

- `choices`: Set of labels
- `gold`: Index of the correct label in choices (Start from 0)

For **Sequential Labeling** tasks (such as [Finer Ord (FinBen_finer_ord)](https://huggingface.co/datasets/TheFinAI/flare-finer-ord)), additional keys should be defined:

- `label`: List of token labels

- `token`: List of tokens

For **Extractive Summarization** tasks (such as [ECTSUM (FinBen_ectsum)](https://huggingface.co/datasets/TheFinAI/flare-ectsum)), additional keys should be defined:

- `label`: List of sentence labels

For **abstractive Summarization** and **Question Answering** tasks (such as [EDTSUM (FinBen_edtsum)](https://huggingface.co/datasets/TheFinAI/flare-edtsum)), no additional keys should be defined

#### Implementing the task

Once your dataset is ready, you can start implementing your task. Your task should be defined within a new class in flare.py or any other Python file located within the tasks directory.

To cater to a range of tasks, we offer several specialized base classes, including `Classification`, `SequentialLabeling`, `RelationExtraction`, `ExtractiveSummarization`, `AbstractiveSummarization` and `QA`.

For instance, if you are embarking on a classification task, you can directly leverage our `Classification` base class. This class allows for efficient and intuitive task creation. To better demonstrate this, let's delve into an example of crafting a task named FinBen-FPB using the `Classification` base class:

```python
class flareFPB(Classification):
    DATASET_PATH = "flare-fpb"
```

And that's it! Once you've created your task class, the next step is to register it in the `src/tasks/__init__.py` file. To do this, add a new line following the format `"task_name": module.ClassName`. Here is how it's done:

```python
TASK_REGISTRY = {
    "flare_fpb": flare.FPB,
    "your_new_task": your_module.YourTask,  # This is where you add your task
}
```

#### Predefined task metrics

| Task                                     | Metric                                 | Illustration                                                 |
| ---------------------------------------- | -------------------------------------- | ------------------------------------------------------------ |
| Classification                           | Accuracy                               | This metric represents the ratio of correctly predicted observations to total observations. It is calculated as (True Positives + True Negatives) / Total Observations. |
| Classification                           | F1 Score                               | The F1 Score represents the harmonic mean of precision and recall, thereby creating an equilibrium between these two factors. It proves particularly useful in scenarios where one factor bears more significance than the other. The score ranges from 0 to 1, with 1 signifying perfect precision and recall, and 0 indicating the worst case. Furthermore, we provide both 'weighted' and 'macro' versions of the F1 score. |
| Classification                           | Missing Ratio                          | This metric calculates the proportion of responses where no options from the given choices in the task are returned. |
| Classification                           | Matthews Correlation Coefficient (MCC) | The MCC is a metric that assesses the quality of binary classifications, producing a score ranging from -1 to +1. A score of +1 signifies perfect prediction, 0 denotes a prediction no better than random chance, and -1 indicates a completely inverse prediction. |
| Sequential Labeling                      | F1 score                               | In the context of Sequential Labeling tasks, we utilize the F1 Score as computed by the `seqeval` library, a robust entity-level evaluation metric. This metric mandates an exact match of both the entity's span and type between the predicted and ground truth entities for a correct evaluation. True Positives (TP) represent correctly predicted entities, False Positives (FP) denote incorrectly predicted entities or entities with mismatched spans/types, and False Negatives (FN) signify missed entities from the ground truth. Precision, recall, and F1-score are then computed using these quantities, with the F1 Score representing the harmonic mean of precision and recall. |
| Sequential Labeling                      | Label F1 score                         | This metric evaluates model performance based solely on the correctness of the labels predicted, without considering entity spans. |
| Relation Extraction                      | Precision                              | Precision measures the proportion of correctly predicted relations out of all predicted relations. It is calculated as the number of True Positives (TP) divided by the sum of True Positives and False Positives (FP). |
| Relation Extraction                      | Recall                                 | Recall measures the proportion of correctly predicted relations out of all actual relations. It is calculated as the number of True Positives (TP) divided by the sum of True Positives and False Negatives (FN). |
| Relation Extraction                      | F1 score                               | The F1 Score is the harmonic mean of precision and recall, and it provides a balance between these two metrics. The F1 Score is at its best at 1 (perfect precision and recall) and worst at 0. |
| Extractive and Abstractive Summarization | Rouge-N                                | This measures the overlap of N-grams (a contiguous sequence of N items from a given sample of text) between the system-generated summary and the reference summary. 'N' can be 1, 2, or more, with ROUGE-1 and ROUGE-2 being commonly used to assess unigram and bigram overlaps respectively. |
| Extractive and Abstractive Summarization | Rouge-L                                | This metric evaluates the longest common subsequence (LCS) between the system and the reference summaries. LCS takes into account sentence level structure similarity naturally and identifies longest co-occurring in-sequence n-grams automatically. |
| Question Answering                       | EmACC                                  | EMACC assesses the exact match between the model-generated response and the reference answer. In other words, the model-generated response is considered correct only if it matches the reference answer exactly, word-for-word. |

>  Additionally, you can determine if the labels should be lowercased during the matching process by specifying `LOWER_CASE` in your class definition. This is pertinent since labels are matched based on their appearance in the generated output. For tasks like examinations where the labels are a specific set of capitalized letters such as 'A', 'B', 'C', this should typically be set to False.

---

## FIT: Financial Instruction Dataset

Our instruction dataset is uniquely tailored for the domain-specific LLM, FinMA. This dataset has been meticulously assembled to fine-tune our model on a diverse range of financial tasks. It features publicly available multi-task and multi-modal data derived from the multiple open released financial datasets.

The dataset is multi-faceted, featuring tasks including sentiment analysis, news headline classification, named entity recognition, question answering, and stock movement prediction. It covers both textual and time-series data modalities, offering a rich variety of financial data. The task specific instruction prompts for each task have been carefully degined by domain experts.

### Modality and Prompts

The table below summarizes the different tasks, their corresponding modalities, text types, and examples of the instructions used for each task:

| **Task**                     | **Modalities**    | **Text Types**        | **Instructions Examples**                                    |
| ---------------------------- | ----------------- | --------------------- | ------------------------------------------------------------ |
| Sentiment Analysis           | Text              | news headlines,tweets | "Analyze the sentiment of this statement extracted from a financial news article.Provide your answer as either negative, positive or neutral. For instance, 'The company's stocks plummeted following the scandal.' would be classified as negative." |
| News Headline Classification | Text              | News Headlines        | "Consider whether the headline mentions the price of gold. Is there a Price or Not in the gold commodity market indicated in the news headline? Please answer Yes or No." |
| Named Entity Recognition     | Text              | financial agreements  | "In the sentences extracted from financial agreements in U.S. SEC filings, identify the named entities that represent a person ('PER'), an organization ('ORG'), or a location ('LOC'). The required answer format is: 'entity name, entity type'. For instance, in 'Elon Musk, CEO of SpaceX, announced the launch from Cape Canaveral.', the entities would be: 'Elon Musk, PER; SpaceX, ORG; Cape Canaveral, LOC'" |
| Question Answering           | Text              | earnings reports      | "In the context of this series of interconnected finance-related queries and the additional information provided by the pretext, table data, and post text from a company's financial filings, please provide a response to the final question. This may require extracting information from the context and performing mathematical calculations. Please take into account the information provided in the preceding questions and their answers when formulating your response:" |
| Stock Movement Prediction    | Text, Time-Series | tweets, Stock Prices  | "Analyze the information and social media posts to determine if the closing price of *\{tid\}* will ascend or descend at *\{point\}*. Please respond with either Rise or Fall." |

### Dataset Statistics

The dataset contains a vast amount of instruction data samples (136K), allowing FinMA to capture the nuances of the diverse financial tasks. The table below provides the statistical details of the instruction dataset:

| Data      | Task                         | Raw    | Instruction | Data Types                | Modalities        | License      | Original Paper |
| --------- | ---------------------------- | ------ | ----------- | ------------------------- | ----------------- | ------------ | -------------- |
| FPB       | sentiment analysis           | 4,845  | 48,450      | news                      | text              | CC BY-SA 3.0 | [1]            |
| FiQA-SA   | sentiment analysis           | 1,173  | 11,730      | news headlines, tweets    | text              | Public       | [2]            |
| Headline  | news headline classification | 11,412 | 11,412      | news headlines            | text              | CC BY-SA 3.0 | [3]            |
| NER       | named entity recognition     | 1,366  | 13,660      | financial agreements      | text              | CC BY-SA 3.0 | [4]            |
| FinQA     | question answering           | 8,281  | 8,281       | earnings reports          | text, table       | MIT License  | [5]            |
| ConvFinQA | question answering           | 3,892  | 3,892       | earnings reports          | text, table       | MIT License  | [6]            |
| BigData22 | stock movement prediction    | 7,164  | 7,164       | tweets, historical prices | text, time series | Public       | [7]            |
| ACL18     | stock movement prediction    | 27,053 | 27,053      | tweets, historical prices | text, time series | MIT License  | [8]            |
| CIKM18    | stock movement prediction    | 4,967  | 4,967       | tweets, historical prices | text, time series | Public       | [9]            |

1. Pekka Malo, Ankur Sinha, Pekka Korhonen, Jyrki Wallenius, and Pyry Takala. 2014. Good debt or bad debt: Detecting semantic orientations in economic texts. Journal of the Association for Information Science and Technology 65, 4 (2014), 782–796.
2. Macedo Maia, Siegfried Handschuh, André Freitas, Brian Davis, Ross McDermott, Manel Zarrouk, and Alexandra Balahur. 2018. Www’18 open challenge: financial opinion mining and question answering. In Companion proceedings of the the web conference 2018. 1941–1942
3. Ankur Sinha and Tanmay Khandait. 2021. Impact of news on the commodity market: Dataset and results. In Advances in Information and Communication: Proceedings of the 2021 Future of Information and Communication Conference (FICC), Volume 2. Springer, 589–601
4. Julio Cesar Salinas Alvarado, Karin Verspoor, and Timothy Baldwin. 2015. Domain adaption of named entity recognition to support credit risk assessment. In Proceedings of the Australasian Language Technology Association Workshop 2015. 84–90.
5. Zhiyu Chen, Wenhu Chen, Charese Smiley, Sameena Shah, Iana Borova, Dylan Langdon, Reema Moussa, Matt Beane, Ting-Hao Huang, Bryan R Routledge, et al . 2021. FinQA: A Dataset of Numerical Reasoning over Financial Data. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing. 3697–3711.
6. Zhiyu Chen, Shiyang Li, Charese Smiley, Zhiqiang Ma, Sameena Shah, and William Yang Wang. 2022. Convfinqa: Exploring the chain of numerical reasoning in conversational finance question answering. arXiv preprint arXiv:2210.03849 (2022).
7. Yejun Soun, Jaemin Yoo, Minyong Cho, Jihyeong Jeon, and U Kang. 2022. Accurate Stock Movement Prediction with Self-supervised Learning from Sparse Noisy Tweets. In 2022 IEEE International Conference on Big Data (Big Data). IEEE, 1691–1700.
8. Yumo Xu and Shay B Cohen. 2018. Stock movement prediction from tweets and historical prices. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 1970–1979.
9. Huizhe Wu, Wei Zhang, Weiwei Shen, and Jun Wang. 2018. Hybrid deep sequential modeling for social text-driven stock prediction. In Proceedings of the 27th ACM international conference on information and knowledge management. 1627–1630.

### Generating Datasets for FIT

When you are working with the Financial Instruction Dataset (FIT), it's crucial to follow the prescribed format for training and testing models.

The format should look like this:

```json
{
    "id": "unique id",
    "conversations": [
        {
            "from": "human",
            "value": "Your prompt and text"
        },
        {
            "from": "agent",
            "value": "Your answer"
        }
    ],
    "text": "Text to be classified",
    "label": "Your label"
}
```

Here's what each field means:

- "id": a unique identifier for each example in your dataset.
- "conversations": a list of conversation turns. Each turn is represented as a dictionary, with "from" representing the speaker, and "value" representing the text spoken in the turn.
- "text": the text to be classified.
- "label": the ground truth label for the text.


The first turn in the "conversations" list should always be from "human", and contain your prompt and the text. The second turn should be from "agent", and contain your answer.

---

## FinMA v0.1: Financial Large Language Model

We are pleased to introduce the first version of FinMA, including three models FinMA-7B, FinMA-7B-full, FinMA-30B, fine-tuned on LLaMA 7B and LLaMA-30B. FinMA-7B and FinMA-30B are trained with the NLP instruction data, while FinMA-7B-full is trained with the full instruction data from FIT covering both NLP and prediction tasks. 

FinMA v0.1 is now available on [Huggingface](https://huggingface.co/TheFinAI/finma-7b-nlp) for public use. We look forward to the valuable contributions that this initial version will make to the financial NLP field and encourage users to apply it to various financial tasks and scenarios. We also invite feedback and shared experiences to help improve future versions.

### How to fine-tune a new large language model using PIXIU based on FIT?

Coming soon.

---

## FinMem: A Performance-Enhanced LLM Trading Agent

FinMem is a novel LLM-based agent framework devised for financial decision-making, encompasses three core modules: Profiling, to outline the agent's characteristics; Memory, with layered processing, to aid the agent in assimilating realistic hierarchical financial data; and Decision-making, to convert insights gained from memories into investment decisions. Currently, FinMem can trade single stocks with high returns after a simple mode warm-up. Below is a quick start for a dockerized version framework, with TSLA as sample input.

Step 1: Set environmental variables
in `.env` add HUGGINGFACE TOKEN and OPENAI API KEY as needed.
```bash
OPENAI_API_KEY = "<Your OpenAI Key>"
HF_TOKEN = "<Your HF token>"
```

Step 2: Set endpoint URL in `config.toml`
Use endpoint URL to deploy models based on the model of choice (OPENAI, Gemini, open source models on HuggingFace, etc.). For open-source models on HuggingFace, one choice for generating TGI endpoints is through RunPod. 
```bash
[chat]
model = "tgi"
end_point = "<set the your endpoint address>"
tokenization_model_name = "<model name>"
...
```

Step 3: Build Docker Image and Container
```bash
docker build -t test-finmem .devcontainer/. 
```
start container:
```bash
docker run -it --rm -v $(pwd):/finmem test-finmem bash
```

Step 4: Start Simulation!
```bash
 Usage: run.py sim [OPTIONS]                                                                                                                
                                                                                                                                            
 Start Simulation                                                                                                                           
                                                                                                                                            
╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --market-data-path    -mdp      TEXT  The environment data pickle path [default: data/06_input/subset_symbols.pkl]                       │
│ --start-time          -st       TEXT  The training or test start time [default: 2022-06-30 For Ticker 'TSLA']                                                               │
│ --end-time            -et       TEXT  The training or test end time [default: 2022-10-11]                                                                 │
│ --run-model           -rm       TEXT  Run mode: train or test [default: train]                                                           │
│ --config-path         -cp       TEXT  config file path [default: config/config.toml]                                                     │
│ --checkpoint-path     -ckp      TEXT  The checkpoint save path [default: data/10_checkpoint_test]                                             │
│ --result-path         -rp       TEXT  The result save path [default: data/11_train_result]                                               │
│ --trained-agent-path  -tap      TEXT  Only used in test mode, the path of trained agent [default: None. Can be changed to data/05_train_model_output OR data/06_train_checkpoint]                                  │
│ --help                                Show this message and exit.                                                                        │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
                              
```
Example Usage:
```bash
python run.py sim --market-data-path data/03_model_input/tsla.pkl --start-time 2022-06-30 --end-time 2022-10-11 --run-model train --config-path config/tsla_tgi_config.toml --checkpoint-path data/06_train_checkpoint --result-path data/05_train_model_output
```

There are also checkpoint functionalities. For more details please visit [FinMem Repository](https://github.com/pipiku915/FinMem-LLM-StockTrading) directly. 

---

## Citation

If you use PIXIU in your work, please cite our paper.

```
@misc{xie2023pixiu,
      title={PIXIU: A Large Language Model, Instruction Data and Evaluation Benchmark for Finance}, 
      author={Qianqian Xie and Weiguang Han and Xiao Zhang and Yanzhao Lai and Min Peng and Alejandro Lopez-Lira and Jimin Huang},
      year={2023},
      eprint={2306.05443},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

@misc{xie2024FinBen,
      title={The FinBen: An Holistic Financial Benchmark for Large Language Models}, 
      author={Qianqian Xie and Weiguang Han and Zhengyu Chen and Ruoyu Xiang and Xiao Zhang and Yueru He and Mengxi Xiao and Dong Li and Yongfu Dai and Duanyu Feng and Yijing Xu and Haoqiang Kang and Ziyan Kuang and Chenhan Yuan and Kailai Yang and Zheheng Luo and Tianlin Zhang and Zhiwei Liu and Guojun Xiong and Zhiyang Deng and Yuechen Jiang and Zhiyuan Yao and Haohang Li and Yangyang Yu and Gang Hu and Jiajia Huang and Xiao-Yang Liu and Alejandro Lopez-Lira and Benyou Wang and Yanzhao Lai and Hao Wang and Min Peng and Sophia Ananiadou and Jimin Huang},
      year={2024},
      eprint={2402.12659},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## License

PIXIU is licensed under [MIT]. For more details, please see the [MIT](LICENSE) file.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=The-FinAI/PIXIU&type=Date)](https://star-history.com/#The-FinAI/PIXIU&Date)
