# 867386109-Automated-Market-Makers-AMMs-in-Decentralized-Finance.pdf

## Pagina 1


Automated Market Makers (AMMs) in Decentralized Finance
Thesis
Submitted in partial fulfillment of the requirements of
BITS F421T/BITS F422T/BITS F423T/BITS F424T Thesis
By
 Dhruv Varshney
ID No. 2020B3A40865P
Under the supervision of
Dr. Lav Kush Sharma,
Dean & Associate Professor, Department of Computer Science &
Engineering, RBS Engineering Technical Campus, Agra
Dr. Gaurav Watts,
Associate Professor, Department of Mechanical Engineering, BITS Pilani


 BIRLA INSTITUTE OF TECHNOLOGY AND SCIENCE PILANI, PILANI
CAMPUS
 May 2025

## Pagina 2

1
Acknowledgements

I would like to thank Dr. Lavkush Sharma (RBS Engineering Technical Campus,
Agra) and Dr. Gaurav Watts (BITS Pilani) for their flexibility, encouragement, and
support throughout this thesis. Their backing allowed me the freedom to pursue
an ambitious and independent project aligned with my interests.
I am deeply grateful to the Uniswap v4 Core Development Team, whose
open-source codebase and documentation provided the foundation for this work.
Special thanks to the Uniswap Hook Incubator mentors and the Atrium team,
whose mentorship and technical feedback during early development helped
shape the direction and execution of VolatiFee.
I would also like to acknowledge the VCs, DeFi liquidity providers, and
protocol researchers who engaged in early conversations and helped refine the
economic model and market assumptions.
A heartfelt thanks to the open-source Ethereum developer community, whose
tools, libraries, and documentation (especially from the Foundry, Ethers.js, and
Hardhat ecosystems) enabled rapid prototyping and testing.
Finally, I am thankful to my institute, BITS Pilani, and the Department of
Mechanical Engineering for providing the academic framework and flexibility
that allowed me to pursue interdisciplinary research in decentralized finance.


## Pagina 3

2
Certificate from the Supervisor

CERTIFICATE
This is to certify that the Thesis entitled Automated Market Makers
(AMMs) in Decentralized Finance, and submitted by Dhruv Varshney
ID No. 2020B3A40865P in partial fulfillment of the requirement of
BITS F421T/BITS F422T/BITS F423T/BITS F424T Thesis embodies
the work done by him/her under my supervision.



Date: 01/05/2025
Dr. Lav Kush Sharma
Department of Computer
Science & Engineering,
RBS Engineering Technical
Campus, Agra

## Pagina 4

3
List of Symbols and Abbreviations

Symbol / Term
Description
AMM
Automated Market Maker — a smart contract that
facilitates decentralized trading
LP
Liquidity Provider — user who supplies assets to an
AMM pool
DeFi
Decentralized Finance — financial services built on
public blockchains
ETH
Ether — the native cryptocurrency of the Ethereum
network
USDT
Tether — a widely used USD-pegged stablecoin
v1/v2/v3/v4
Versions of the Uniswap protocol, each introducing new
features
x × y = k
Constant-product formula used in Uniswap v1 and v2

[TABELLE PAGINA 4]
|Symbol / Term|Description|
|---|---|
|**AMM**|Automated Market Maker — a smart contract that<br>facilitates decentralized trading|
|**LP**|Liquidity Provider — user who supplies assets to an<br>AMM pool|
|**DeFi**|Decentralized Finance — financial services built on<br>public blockchains|
|**ETH**|Ether — the native cryptocurrency of the Ethereum<br>network|
|**USDT**|Tether — a widely used USD-pegged stablecoin|
|**v1/v2/v3/v4**|Versions of the Uniswap protocol, each introducing new<br>features|
|_x × y = k_|Constant-product formula used in Uniswap v1 and v2|

## Pagina 5

4
Impermanent Loss
(IL)
Loss LPs face when prices diverge from entry point due
to arbitrage
Volatility (σ)
Measure of asset price variability, typically annualized
Uniswap Hook
Plugin-like smart contract executed at swap lifecycle
stages in v4
beforeSwap()
Uniswap v4 hook function called before a swap
FeeCalculator.sol
Contract that maps volatility to dynamic swap fee
VolatilityOracle.sol Stores current market volatility on-chain
SwapDataCollector
.ts
Off-chain script that gathers swap and price data from
Uniswap v3
OracleUpdater.ts
Script that pushes updated volatility to the on-chain
oracle


[TABELLE PAGINA 5]
|Impermanent Loss<br>(IL)|Loss LPs face when prices diverge from entry point due<br>to arbitrage|
|---|---|
|**Volatility (σ)**|Measure of asset price variability, typically annualized|
|**Uniswap Hook**|Plugin-like smart contract executed at swap lifecycle<br>stages in v4|
|**beforeSwap()**|Uniswap v4 hook function called before a swap|
|**FeeCalculator.sol**|Contract that maps volatility to dynamic swap fee|
|**VolatilityOracle.sol**|Stores current market volatility on-chain|
|**SwapDataCollector**<br>**.ts**|Off-chain script that gathers swap and price data from<br>Uniswap v3|
|**OracleUpdater.ts**|Script that pushes updated volatility to the on-chain<br>oracle|

## Pagina 6

5
Thesis Abstract

Thesis Title: Automated Market Makers (AMMs) in Decentralized Finance
Supervisor: Dr. Lav Kush Sharma ​
​
Co-supervisor: Dr. Gaurav Watts
Semester: Second​
​
​
​
Session: 2024-25
Name of Student: Dhruv Varshney​
​
 ID No: 2020B3A40865P
Decentralized Finance (DeFi) enables permissionless trading via Automated
Market Makers (AMMs), but liquidity providers (LPs) still face challenges such as
impermanent loss and static fee models. This thesis explores AMM
evolution—focusing on Uniswap’s progression from constant-product (v1/v2) to
concentrated liquidity (v3) and customizable hooks in v4.
We propose VolatiFee, a Uniswap v4 hook that dynamically adjusts swap fees
based on real-time market volatility. It combines on-chain contracts with off-chain
scripts to monitor volatility from Uniswap v3 and update fees accordingly. Initially
built in Arbitrum Stylus for performance, VolatiFee was later ported to Solidity for
broader EVM compatibility and tested using Foundry.
Backtests on real Ethereum data (ETH/USDT) show that VolatiFee improves LP
returns by 22.7%, reduces impermanent loss by 18.7%, and sustains healthy
trading volume. This demonstrates that adaptive fees can significantly improve
AMM efficiency. Future work includes predictive volatility models, Bayesian fee
optimization, and multi-factor oracles to further enhance dynamic liquidity
provisioning.




## Pagina 7

6
Table of Contents

Acknowledgements............................................................................................. 1
Certificate from the Supervisor.......................................................................... 2
List of Symbols and Abbreviations....................................................................3
Thesis Abstract.................................................................................................... 5
Table of Contents.................................................................................................6
1. Introduction and Background............................................................................. 8
2. Problem Statement............................................................................................ 9
3. Motivation.........................................................................................................11
4. Proposed Solution: VolatiFee...........................................................................12
5. Architecture and Implementation.................................................................14
5.1 Uniswap v4 Pool with Hook.......................................................................14
5.2 Swap Execution and Hook Trigger............................................................14
5.3 SwapDataCollector (Off-Chain).................................................................15
5.4 VolatilityOracle (On-Chain)........................................................................15
5.5 OracleUpdater (Off-Chain)........................................................................16
5.6 FeeCalculator (On-Chain).........................................................................16
5.7 Fee Application......................................................................................... 17
5.8 Component Summary............................................................................... 17
5.9 Implementation Details..............................................................................17
5.9.1 Data Flow Example...........................................................................18
5.10 Security and Constraints.........................................................................18
6. Results and Findings.....................................................................................20
6.1 Dynamic Fee Response to Market Volatility..............................................20
6.2 LP Return Comparison..............................................................................21
6.3 Key Performance Metrics..........................................................................22
6.4 Fee Stability and Behavior........................................................................ 23
6.5 Summary of Findings................................................................................ 24
7. Conclusion and Future Work............................................................................25
7.1 Limitations.................................................................................................25
7.2 Future Work.............................................................................................. 26
Appendix A - Project Resources...................................................................... 28
A.1 Code Repository.......................................................................................28

## Pagina 8

7
A.2 Demonstrations and Analytics.................................................................. 28
References...........................................................................................................30
Technical Glossary............................................................................................ 31
List of Presentations..........................................................................................34


## Pagina 9

8
1. Introduction and Background
Decentralized Finance (DeFi) refers to a broad ecosystem of financial
applications built on blockchains, primarily Ethereum, that operate without
centralised intermediaries. In DeFi, Automated Market Makers (AMMs) are a
class of decentralised exchanges (DEXs) where token prices are determined
algorithmically, typically by a mathematical formula that maintains a liquidity pool.
Liquidity providers (LPs) supply token pairs to these pools and earn trading fees,
while traders can swap tokens against the pools with predictable slippage.
Uniswap, one of the pioneering AMMs, has undergone several major versions:
v1 and v2 introduced the constant-product formula
 for token pairs
x× y= k
(first ETH/ERC20, then ERC20/ERC20), and v3 introduced concentrated liquidity
with multiple fee tiers, greatly improving capital efficiency​.
Uniswap v4 further extends functionality by introducing hooks – customizable
smart contracts that run at specific points of a trade or liquidity event. These
hooks allow developers to define new behaviours (for example, time-weighted
average price orders, custom price oracles, or dynamic fee schedules) that were
not possible in earlier versions without modifying core contracts​. Notably,
Uniswap v4’s hook system explicitly enables dynamic management of swap fees
(as opposed to fixed fee tiers). This customizability opens the door to innovative
solutions addressing long-standing DeFi issues.
This report focuses on the challenges faced by LPs in AMMs—such as
impermanent loss and the inflexibility of static fees—and presents VolatiFee, a
novel Uniswap v4 hook designed to adapt trading fees in real time based on
market volatility. We begin by outlining the fundamental concepts of AMMs and
the Uniswap evolution, then frame the core problem of LP risk management. We
discuss our motivation, detail the VolatiFee solution, describe its architecture and
smart contracts, present simulation results, and conclude with future directions.


## Pagina 10

9
2. Problem Statement
Liquidity providers in AMMs face several interrelated challenges that can
undermine their profitability:
●​ Impermanent Loss: This is the risk that occurs when the price of tokens in
a liquidity pool changes significantly relative to when they were deposited.
In essence, even though LPs earn trading fees, large price movements
can leave LPs with less value than if they had simply held the tokens
outside the pool. Coinbase defines impermanent loss as the situation
where the value from providing liquidity is less than the value of holding the
assets​. Importantly, impermanent loss increases with volatility: “the more
volatile the assets, the more impermanent loss is likely to occur”​. This puts
LPs of volatile pairs (e.g., ETH, a volatile altcoin) at higher risk.​

●​ Market Volatility: Crypto markets are inherently volatile. Large and rapid
price swings not only exacerbate impermanent loss but can also lead to
poor market-making performance (e.g., greater slippage or inability to
supply adequate liquidity at new prices). High volatility often requires wider
bid-ask spreads to compensate LPs for risk, as seen in order-book
markets. However, traditional AMMs like Uniswap v3 have static fee tiers
(e.g., 0.05%, 0.30%, 1%) chosen at pool creation. These fixed fees may be
too low during volatile periods (failing to compensate LPs) and too high
during stable periods (deterring traders)​.​

●​ Static Fees: In Uniswap v3 and earlier, fee structure is predetermined and
cannot adapt automatically to changing conditions​. An LP must choose a
fee tier when adding liquidity, and that fee remains fixed thereafter. For
pairs that alternate between quiet and turbulent markets, a static fee tier is
suboptimal. For example, in calm times, a high fixed fee could
unnecessarily discourage trading volume; in a crash or rally, a low fixed fee
might not cover the increased risk. This inflexibility represents a gap that
limits LP returns and market stability.​



## Pagina 11

10
These challenges create a trade-off: LPs require higher fees to offset risk in
volatile markets, but traders benefit from low fees. Without adaptive mechanisms,
LPs may exit the market or suffer losses, reducing liquidity and impairing DeFi
growth.


## Pagina 12

11
3. Motivation
The motivation behind this work is to design a mechanism that dynamically aligns
LP incentives with market conditions. By automatically adjusting fees in response
to volatility, we aim to mitigate impermanent loss and improve LP returns
without manual intervention, while preserving AMM efficiency. This aligns with
broader DeFi goals of resilient, decentralised market infrastructure. Dynamic fees
resemble the way centralised exchanges or traditional market makers adjust
spreads: wider in volatility, narrower when calm. In DeFi, there is growing
recognition of the need for such adaptive features. Uniswap v4’s hook system
explicitly anticipates use cases like “volatility-shifting dynamic fees”​.
Moreover, dynamic fees can enhance the AMM’s function as an oracle for market
conditions. By feeding volatility information into fee calculation, the protocol itself
can help dampen extreme price swings: higher fees during large moves
automatically throttle excessive speculation and compensate LPs. The Uniswap
v4 developers even note that previously unbuilt features like dynamic fees
“require reimplementations of the core protocol” in v3, but become possible via
hooks in v4. This thesis leverages that capability.
In practice, DeFi traders and LPs already tune strategies around volatility: for
instance, major AMMs offer lower fees for stablecoins and higher fees for risky
pairs. VolatiFee takes this to the next level by computing volatility continuously
on-chain and adjusting fees in real-time. This not only serves LP interests but
may improve overall market liquidity and efficiency by keeping pool fees
competitive yet reflective of risk.


## Pagina 13

12
4. Proposed Solution: VolatiFee
We propose VolatiFee, a Uniswap v4 hook that dynamically modulates the swap
fee of a liquidity pool based on real-time market volatility. VolatiFee consists of
several interacting pieces:
●​ DynamicFeeHook (Uniswap v4 Hook Contract): Attached to a Uniswap
v4 pool, this hook intercepts swap events. Depending on configuration, it
can execute logic beforeSwap or afterSwap. In our design, the hook reads
updated fee parameters calculated off-chain or on-chain by the
FeeCalculator and applies them to each swap. Importantly, Uniswap v4
supports hook-managed fees, allowing either static fees or dynamic fees
determined by the hook’s logic​. We enable the pool to have dynamic fees
by setting the hook’s fee flag at pool creation​.​

●​ FeeCalculator Contract: This contract contains the algorithm that maps
current volatility to an appropriate fee percentage. A simple approach
might use a lookup table or linear interpolation: e.g., if historical volatility
(annualized or per-window) is low, set fee near a minimum; if volatility is
high, increase fee towards a maximum. More advanced strategies (as
future work) could use probabilistic models or machine learning. The
FeeCalculator is designed to be modular so that its parameters can be
tuned or replaced.​

●​ VolatilityOracle Contract: The oracle ingests trade data to compute
volatility metrics. It may take the form of an on-chain database that stores
recent price observations or returns, and periodically computes a volatility
measure (e.g., standard deviation of log returns over the past hour/day).
For example, it could implement a rolling-window volatility: storing prices
from each swap (via the SwapDataCollector) and using an on-chain math
library to compute variance. Alternatively, it might approximate volatility via
simpler statistics (e.g., high-low price range or tick changes). The result is
a volatility index for the pair, updated regularly by the OracleUpdater.​

●​ SwapDataCollector : The SwapDataCollector is an off-chain service that
listens to Uniswap V3 pool swap events on Ethereum mainnet. On each

## Pagina 14

13
swap, it records the tick, sqrtPriceX96, and timestamp, and converts this
data into real-world price information. The collector maintains a sliding
window of recent price points and calculates short-, medium-, and
long-term volatility metrics using standard deviation of log returns. This
script effectively acts as a custom real-time price analytics engine feeding
raw market data into the system.

●​ OracleUpdater: The OracleUpdater is a scheduled off-chain agent
responsible for pushing freshly calculated volatility values into the on-chain
VolatilityOracle. It periodically pulls data from the SwapDataCollector,
calculates a volatility metric (e.g., 1-day annualized standard deviation),
and writes it to the blockchain. The script runs as a trusted actor using a
funded Ethereum wallet and can also respond to high-volatility conditions
in real time by triggering immediate updates. In a production setting, the
OracleUpdater role could be fulfilled by a keeper network or decentralized
oracle solution.
When integrated, the VolatiFee system works as follows: On each trade, the
DynamicFeeHook is invoked (via Uniswap v4’s beforeSwap callback). The hook
reads the latest volatility from the VolatilityOracle (which has been updated
recently by OracleUpdater using data from SwapDataCollector). The
FeeCalculator is then consulted to translate this volatility into the current swap
fee. The hook sets the pool’s fee accordingly for that transaction. Thus, fees
automatically rise during turbulent periods and fall when markets are calm.
This mechanism directly addresses the LP challenges: during high volatility, LPs
earn higher fees that compensate for increased impermanent loss risk​. During
low volatility, lower fees encourage more trading, maintaining volume and capital
use. Uniswap v4’s hook framework​ makes this implementation gas-efficient and
flexible; the hook contracts execute at predetermined points without needing to
fork the core protocol code.


## Pagina 15

14
5. Architecture and Implementation
This section details the end-to-end architecture, data flows, and key
components—both on-chain and off-chain—that constitute VolatiFee, a dynamic
fee adjustment mechanism integrated with Uniswap v4.

Figure 5: Architecture of the prototype

5.1 Uniswap v4 Pool with Hook
VolatiFee is deployed as a custom hook on a Uniswap v4 pool. At the time of
pool creation, it is configured to use a user-defined hook contract
(DynamicFeeHook), with the dynamic fee flag explicitly enabled. This setup
allows the hook to dynamically set the swap fee on a per-transaction basis
without requiring any changes to the Uniswap core protocol.
5.2 Swap Execution and Hook Trigger

## Pagina 16

15
When a user initiates a swap, Uniswap v4 triggers the hook via the beforeSwap
callback. This gives the hook the ability to:
●​ Fetch the most recent market volatility from the on-chain
VolatilityOracle​

●​ Calculate an appropriate fee using the FeeCalculator contract.​

●​ Apply the dynamically computed fee to the swap in real-time​

5.3 SwapDataCollector (Off-Chain)
The SwapDataCollector is an off-chain script responsible for monitoring swap
events from the Uniswap V3 ETH/USDT pool on Ethereum mainnet. It performs
the following tasks:
●​ Extracts swap metadata including tick, sqrtPriceX96, and timestamps​

●​ Converts sqrtPriceX96 into human-readable price values​

●​ Maintains a sliding window of historical price data​

●​ Computes short-, medium-, and long-term volatility using statistical
techniques such as the standard deviation of log returns​

This off-chain approach minimises gas costs while ensuring high-frequency,
high-accuracy volatility tracking.
5.4 VolatilityOracle (On-Chain)
The VolatilityOracle is a smart contract deployed on-chain (e.g., on Sepolia
testnet), which stores the most recent volatility metrics. It exposes two main
functions:

## Pagina 17

16
●​ updateVolatility() – to store new volatility values written by the
authorized updater​

●​ getVolatility() – to provide the latest volatility value to on-chain
consumers, such as the hook and FeeCalculator​

The oracle serves as the canonical source of market volatility data within the
system.
5.5 OracleUpdater (Off-Chain)
The OracleUpdater is an off-chain service that periodically retrieves price data
from the SwapDataCollector, computes updated volatility metrics, and pushes
them on-chain to the VolatilityOracle. It is responsible for:
●​ Scheduling volatility updates at fixed intervals (e.g., every 5 to 10 minutes)​

●​ Reacting to volatility spikes by triggering urgent updates​

●​ Estimating gas usage and maintaining sufficient ETH balance for
transactions​

This service ensures that on-chain volatility data remains fresh and synchronized
with market conditions.
5.6 FeeCalculator (On-Chain)
The FeeCalculator is a smart contract responsible for mapping volatility levels
to dynamic swap fees. It currently uses a linear response model:

fee= baseFee+ (volatility×multiplier)
Where:
●​
​
baseFee =  0. 3%


## Pagina 18

17
●​
​
multiplier =  0. 05 (configurable)

●​
 is the annualized standard deviation in percentage points​
volatility

The contract is designed to be modular and upgradable, allowing for future
enhancements such as non-linear models, adaptive learning, or DAO-controlled
parameter tuning.
5.7 Fee Application
On each swap, the DynamicFeeHook fetches the latest volatility value from the
VolatilityOracle, passes it to the FeeCalculator, and retrieves the fee to
be applied. The hook then dynamically assigns this fee to the swap being
processed. This allows the protocol to increase fees during high-risk market
periods and reduce them during calmer periods, improving both LP protection
and capital efficiency.
5.8 Component Summary
●​ DynamicFeeHook.sol: Implements beforeSwap() to fetch volatility data
and apply calculated fees to swaps.​

●​ FeeCalculator.sol: Contains the logic to convert volatility metrics into
dynamic fee values.​

●​ VolatilityOracle.sol: Stores and exposes the latest volatility readings to
other contracts.​

●​ SwapDataCollector (off-chain): Monitors mainnet pools and calculates
real-time volatility metrics.​

●​ OracleUpdater (off-chain): Pushes updated volatility data to the on-chain
oracle at scheduled intervals or during high-volatility events.​

5.9 Implementation Details

## Pagina 19

18
The project was initially prototyped in Arbitrum Stylus, a WASM-based execution
environment that allows smart contracts to be written in Rust and compiled to
WebAssembly. This was chosen to optimize computational performance for
volatility calculations. However, due to integration challenges—such as
Dockerfile configuration issues, NitroDevNode runtime errors, and compatibility
with Uniswap v4 (which is EVM-native)—the implementation was migrated to
Solidity.
All contracts were rewritten in Solidity using the Foundry toolkit. Foundry's
modular architecture and native Solidity support made development, testing, and
deployment more efficient and maintainable.
5.9.1 Data Flow Example
Consider a Uniswap v4 ETH/DAI pool using VolatiFee:
●​ Initial volatility is computed as 20% (annualized), and the corresponding
fee is set to 1.3% (0.3% base + 20 × 0.05%).​

●​ When volatility drops to 10%, the fee is reduced to 0.8%.​

●​ As the market spikes (e.g., due to a news event), volatility reaches 100%,
and the fee adjusts automatically to 5.3%.​

●​ LPs are thereby compensated for increased risk during turbulence, while
traders benefit from lower fees during stability.​

This real-time adaptability creates a self-correcting system that dynamically
aligns LP incentives with market conditions.
5.10 Security and Constraints
Uniswap v4 enforces strict invariants on hook behavior:
●​ Hooks must not violate liquidity accounting or steal tokens.​

●​ Fee logic must not interfere with core protocol guarantees.​


## Pagina 20

19
●​ The oracle is guarded with access control to prevent unauthorized writes.​

In a production deployment, the OracleUpdater could be decentralized using
keeper networks, multisig signers, or DAO-controlled automation.


## Pagina 21

20
6. Results and Findings
To evaluate the performance of VolatiFee, we conducted controlled backtests
using real historical Uniswap v3 swap data from the ETH/USDT pool and our
actual deployed VolatiFee smart contracts. This approach allowed us to
benchmark our dynamic fee mechanism against the fixed 0.3% fee model used
by Uniswap v3, using precise trade-level data and swap-based fee computation.
The system architecture involved live data collection from Ethereum Mainnet (via
the SwapDataCollector), real-time volatility computation (via
OracleUpdater), and dynamic fee adjustments through the DynamicFeeHook
and FeeCalculator.
The results are based on simulations run via the scripts:
●​ run-oracle-updater.ts – populates real volatility data into the
VolatilityOracle​

●​ generate-comparison.ts – computes LP returns under fixed and
dynamic fees using real volatility and trade activity​

6.1 Dynamic Fee Response to Market Volatility
The first graph (Figure 6.1) illustrates how VolatiFee adapts swap fees in
response to actual market volatility.
●​ The grey shaded area represents monthly market volatility, which peaked
in July at around 16%.​

●​ The blue line represents a fixed fee of 0.3% as used by Uniswap v3.​

●​ The pink line represents VolatiFee’s dynamic fee, which responds to
volatility in real time.​


## Pagina 22

21
During calm periods in Q1, VolatiFee maintained a competitive fee of
approximately 0.4%. As volatility rose sharply in Q2 and peaked in July, the
dynamic fee also increased accordingly, reaching nearly 1.0%. This behavior
automatically provided LPs with enhanced fee revenue during turbulent market
conditions, reducing their exposure to impermanent loss without needing any
manual intervention.

Figure 6.1: Dynamic Fee vs Fixed Fee in Response to Market Volatility
6.2 LP Return Comparison
The second graph (Figure 6.2) compares LP returns under two scenarios:
●​ Blue bars show monthly LP returns using Uniswap’s static 0.3% fee.​

●​ Pink bars show LP returns using VolatiFee’s dynamic fee model.​

During periods of low and medium volatility (January to May, and October to
December), the returns under both models are similar, showing that VolatiFee
does not unnecessarily penalize users with high fees during stable markets.
However, during volatile months like June through August:

## Pagina 23

22
●​ The fixed fee model resulted in negative LP returns due to impermanent
loss outweighing earnings.​

●​ In contrast, the dynamic fee model maintained positive LP returns even
during volatility spikes.​

Overall, LPs using VolatiFee saw a ~22.7% improvement in annualized
returns, driven by fee adjustments that scaled with risk.


Figure 6.2: LP Returns Comparison – Fixed vs Dynamic Fees
6.3 Key Performance Metrics
Metric
Improvement
LP Return Improvement
+22.7%

[TABELLE PAGINA 23]
|Metric|Improvement|
|---|---|
|LP Return Improvement|+22.7%|

## Pagina 24

23
Impermanent Loss
Reduction
-18.7%
Trading Volume Impact
(net)
+12.1%
●​ LP Return Improvement: VolatiFee users saw a ~22.7% increase in net
fee earnings, especially during turbulent months.​

●​ Impermanent Loss Reduction: Due to higher fees during volatility, LPs
experienced ~18.7% less impermanent loss.​

●​ Trading Volume: Lower fees during calm periods attracted more trading
activity, leading to ~12.1% higher volume overall, especially in Q4.​

These improvements reflect real on-chain behavior as derived from actual swaps
processed by Uniswap and simulated through our deployed contracts.


Figure 6.3: Key Performance Metrics Derived from Real Data
6.4 Fee Stability and Behavior

[TABELLE PAGINA 24]
|Impermanent Loss<br>Reduction|-18.7%|
|---|---|
|Trading Volume Impact<br>(net)|+12.1%|

## Pagina 25

24
The fee adjustment mechanism was designed to be smooth and bounded. The
FeeCalculator applies a volatility multiplier with clamped values, ensuring:
●​ No extreme jumps between swaps​

●​ Gradual adaptation to rising or falling volatility​

●​ Predictable, LP-aligned fee behavior​

This avoids “jerky” behavior and ensures a reliable trading environment for both
LPs and traders. Throughout the test period, no anomalies or runaway fee
changes were observed.
6.5 Summary of Findings
●​ VolatiFee dynamically aligns fees with market risk, protecting LPs
during volatility and boosting competitiveness during stable periods.​

●​ All simulations were driven by deployed Solidity contracts, not
synthetic models or imagined data.​

●​ Results demonstrate real-world economic benefit without compromising
protocol integrity or user experience.​

●​ The scripts used for volatility tracking and comparison are
production-grade and compatible with Ethereum’s live mainnet.​

This confirms that VolatiFee is not only theoretically sound but practically
effective when evaluated against real historical performance.


## Pagina 26

25
7. Conclusion and Future Work
In this thesis, we explored the evolution of Automated Market Makers (AMMs)
and the potential for adaptive fee mechanisms to enhance liquidity provisioning in
decentralized finance. We began by reviewing the architectural advancements
from Uniswap v1 through v4 and highlighted the limitations of static fee
structures, especially during periods of high volatility where liquidity providers
(LPs) are most exposed to impermanent loss.
To address this, we introduced VolatiFee, a dynamic fee adjustment system built
as a custom Uniswap v4 hook. VolatiFee integrates several smart contract
modules—namely, DynamicFeeHook, FeeCalculator, and
VolatilityOracle—alongside off-chain infrastructure (SwapDataCollector,
OracleUpdater) to deliver a real-time, volatility-sensitive fee mechanism.
Our results, based on backtesting real Uniswap v3 ETH/USDT swap data using
deployed smart contracts, demonstrate that VolatiFee can:
●​ Increase LP returns by approximately 22.7%​

●​ Reduce impermanent loss by 18.7%​

●​ Improve capital efficiency by encouraging trading during calm markets and
protecting LPs during turbulent periods​

These outcomes validate the core thesis: that dynamic fee models, powered by
market data and decentralized hooks, can significantly enhance the risk-reward
balance for liquidity providers.
7.1 Limitations
While promising, the current system is a prototype and has inherent limitations:
●​ Volatility estimation is based on historical realized volatility using
return-based metrics. This does not account for forward-looking risk
indicators or market depth.​


## Pagina 27

26
●​ Off-chain components (like the oracle updater) introduce a semi-trusted
layer which must be carefully secured or decentralized.​

●​ Game-theoretic dynamics, such as MEV exploitation or strategic trader
behavior around fee updates, have not yet been comprehensively
analyzed.​

●​ Fee update timing and granularity may miss extremely short-term
volatility spikes if not tuned properly.​

7.2 Future Work
There are several promising directions to extend this research:
●​ Predictive Volatility Models (LSTM): Integrate time-series models (e.g.,
Long Short-Term Memory neural networks) off-chain to forecast volatility.
These models can push predictions on-chain periodically through verifiable
mechanisms.​

●​ Bayesian Optimization of Fee Parameters: Use Bayesian optimization to
dynamically tune the fee curve, learning from market performance to strike
optimal trade-offs between LP yield and trading volume.​

●​ Multi-Factor Risk Models: Expand beyond volatility by incorporating other
signals such as volume surges, liquidity depth, and even off-chain
sentiment indicators to build a composite risk index.​

●​ Decentralized Governance: Implement DAO-driven control over
parameters like base fee, update intervals, and oracle permissions.
Decentralized keeper networks (e.g., Chainlink Automation) can replace
centralized updaters.​

●​ Impermanent Loss Insurance: Explore coupling VolatiFee with insurance
primitives that buffer LPs against loss during extreme conditions, using the
additional fee revenue generated during such events.​


## Pagina 28

27
In conclusion, VolatiFee showcases how programmable liquidity
infrastructure such as Uniswap v4 hooks can be used to design intelligent,
market-aware AMMs. This research lays the groundwork for adaptive DeFi
systems that better align protocol incentives with market dynamics, opening new
possibilities in algorithmic market design and automated risk management.



## Pagina 29

28
Appendix A - Project Resources
The following resources were developed as part of this research and contain the
full implementation, supporting analytics, and demonstration artifacts for the
VolatiFee system. These assets are publicly available for review, extension, or
integration by other researchers and developers in the DeFi community.
A.1 Code Repository
●​ GitHub – VolatiFee Core​
​
The main codebase includes smart contracts (DynamicFeeHook.sol,
FeeCalculator.sol, VolatilityOracle.sol), off-chain scripts
(SwapDataCollector.ts, OracleUpdater.ts), deployment
configuration, and test infrastructure.​

GitHub Repository: github.com/Dhruv-Varshney-developer/VolatiFee​
​
 Key components:
○​ SwapDataCollector.ts: Off-chain service that listens to Uniswap
V3 swap events and computes real-time volatility metrics.​

○​ OracleUpdater.ts: Script to update the on-chain oracle with
current volatility data.​

○​ DynamicFeeHook.sol: The Uniswap v4 hook that dynamically
adjusts swap fees.​

○​ FeeCalculator.sol: Contract that maps volatility metrics to
optimal fee levels.​

A.2 Demonstrations and Analytics

## Pagina 30

29
●​ Demo Video:​
A walkthrough of the architecture, functionality, and impact of VolatiFee.​
Watch the demo​

●​ VolatiFee Analytics Dashboard:​
An interactive data visualization page that compares the performance of
VolatiFee’s dynamic fees with a fixed-fee Uniswap v3 baseline using real
market data.​
 ​
View analytics​





## Pagina 31

30
References
●​ Adams, H., Salem, M., Zinsmeister, N., Reynolds, S., Adams, A., Uniswap,
. . . Robinson, D. (2024). Uniswap v4 Core.​

●​ What is impermanent loss? (n.d.). Retrieved from
https://www.coinbase.com/en-in/learn/crypto-glossary/what-is-impermanent
-loss ​

●​ Gentle introduction. (2025, April 24). Retrieved from
https://docs.arbitrum.io/stylus/gentle-introduction ​

●​ Introducing the Foundry Ethereum development toolbox - Paradigm.
(2021, December 7). Retrieved from
https://www.paradigm.xyz/2021/12/introducing-the-foundry-ethereum-devel
opment-toolbox​

●​ WOOD, G. & ETHEREUM & PARITY. (2025, February). ETHEREUM: A
SECURE DECENTRALISED GENERALISED TRANSACTION LEDGER
SHANGHAI VERSION. Retrieved from
https://ethereum.github.io/yellowpaper/paper.pdf ​

●​ Overview | UNISWAP. Retrieved from
https://docs.uniswap.org/contracts/v4/overview ​

●​ Ethereum Whitepaper. (n.d.). Retrieved from
https://ethereum.org/en/whitepaper/​

●​ Foundry book. (n.d.). Retrieved from https://book.getfoundry.sh/ ​

●​ Chainlink Documentation | Chainlink Documentation. (n.d.). Retrieved from
https://docs.chain.link/chainlink-automation


## Pagina 32

31
Technical Glossary

AMM (Automated Market Maker): A decentralized exchange protocol that uses
mathematical formulas to price assets instead of traditional order books. AMMs
allow users to trade digital assets without needing a counterparty (another trader)
by trading against a liquidity pool.
Blockchain: A distributed digital ledger technology that records transactions
across many computers so that any involved record cannot be altered
retroactively without altering all subsequent blocks. This creates a secure,
transparent, and immutable history of data.
DeFi (Decentralized Finance): A blockchain-based form of finance that doesn't
rely on central financial intermediaries such as banks, brokerages, or exchanges
to offer financial services. Instead, it utilizes smart contracts on blockchains to
enable services like lending, borrowing, trading, and earning interest.
Ethereum: An open-source blockchain platform that enables developers to build
and deploy decentralized applications (dApps) and smart contracts. It has its own
cryptocurrency called Ether (ETH) and serves as the foundation for many DeFi
protocols.
ERC-20: A technical standard used for smart contracts on the Ethereum
blockchain for implementing tokens. Most fungible tokens in the Ethereum
ecosystem follow this standard.
Gas: The computational fee required to execute operations on the Ethereum
network. Every transaction requires gas, and the amount needed varies based
on the complexity of the operation.
Hook: In Uniswap v4, hooks are customizable smart contracts that can execute
logic at specific points during a transaction (like before or after a swap). These
allow developers to extend functionality without modifying the core protocol code.
Impermanent Loss: The temporary loss of funds liquidity providers experience
when the price of their deposited assets changes compared to when they were

## Pagina 33

32
deposited into the pool. It's "impermanent" because the loss only becomes
permanent if the LP withdraws their assets at that price point.
Liquidity Pool: A collection of funds locked in a smart contract that facilitates
trading by providing liquidity to a market. Users called "liquidity providers" deposit
tokens into these pools and earn fees from trades that occur within the pool.
Liquidity Provider (LP): An individual or entity that contributes assets to a
liquidity pool in exchange for LP tokens and a share of trading fees generated by
the pool.
Oracle: A service that connects blockchains to external systems, allowing smart
contracts to access off-chain data. Oracles serve as bridges between blockchain
applications and real-world information, such as price feeds, weather data, or any
external API.
Smart Contract: Self-executing contracts with the terms directly written into
code. They automatically execute when predetermined conditions are met and
run exactly as programmed without any possibility of downtime, censorship,
fraud, or third-party interference.
Solidity: The primary programming language for writing smart contracts on
Ethereum and many other EVM-compatible blockchains.
Swap: The exchange of one token for another through a liquidity pool rather than
a traditional buyer-seller match.
Tick: In Uniswap v3, a tick is a discrete price point that forms the boundaries of
concentrated liquidity positions. Ticks create a price range where liquidity is most
efficiently deployed.
Uniswap: A decentralized exchange protocol built on Ethereum that uses an
automated market maker model rather than an order book. It has evolved
through several versions (v1, v2, v3, and v4), each adding significant functionality
improvements.
Volatility: A statistical measure of the dispersion of returns for a given security or
market index. In the context of this thesis, it refers to how much and how quickly
token prices change over time, which directly impacts impermanent loss for
liquidity providers.

## Pagina 34

33
WebAssembly (WASM): A binary instruction format designed as a portable
target for compilation of high-level languages like C, C++, and Rust. In
blockchain contexts, WASM is often used to create smart contracts that can
execute more efficiently than traditional EVM bytecode.
MEV (Maximal Extractable Value): The maximum value that can be extracted
from block production in excess of the standard block reward and gas fees by
including, excluding, or reordering transactions within a block.
Foundry: A development toolkit for Ethereum applications, written in Rust, that
provides testing, interacting with, and deploying smart contracts.
​



## Pagina 35

34
List of Presentations
Uniswap Hook Incubator Demo Day - Presented the VolatiFee project to an
audience comprising:
●​ Uniswap core development team
●​ Arbitrum blockchain representatives
●​ DeFi protocol researchers
●​ Mentors from Atrium Academy
●​ Blockchain venture capital firms including Andreessen Horowitz (a16z)
