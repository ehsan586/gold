//+------------------------------------------------------------------+
//| GoldGuardEA.mq5 - independent last-line-of-defence inside MT5.    |
//| It NEVER opens trades. It only protects the account even if the  |
//| Python backend crashes or hangs:                                  |
//|  1. refuses to run on a REAL account (unless explicitly allowed)  |
//|  2. adds an emergency stop-loss to any bot position without SL    |
//|  3. closes bot positions that exceed the hard lot cap             |
//|  4. HARD KILL: if equity falls InpMaxDrawdownPct from its peak,   |
//|     closes all bot positions and keeps closing new ones (HALT)    |
//| NOT COMPILED/TESTED by the developer: compile in MetaEditor (F7)  |
//| and test on a DEMO account first.                                  |
//+------------------------------------------------------------------+
#property copyright "goldbot"
#property version   "1.00"
#property strict
#include <Trade/Trade.mqh>

input long   InpMagic            = 20261002;  // must equal GOLDBOT_MAGIC_NUMBER
input double InpMaxDrawdownPct   = 15.0;      // hard kill threshold (should be above the Python limit)
input double InpMaxLot           = 0.50;      // closes any bot position larger than this
input int    InpEmergencySLPts   = 1500;      // points (gold: 1500 = 15.00 USD) when SL is missing
input bool   InpAllowReal        = false;     // keep false
input bool   InpResetHalt        = false;     // set true once to clear a HALT after you reviewed things

CTrade  trade;
string  GV_PEAK = "GG_PEAK";
string  GV_HALT = "GG_HALT";

int OnInit()
{
   if(AccountInfoInteger(ACCOUNT_TRADE_MODE) == ACCOUNT_TRADE_MODE_REAL && !InpAllowReal)
   {
      Alert("GoldGuardEA: REAL account detected - refusing to run (demo only).");
      return INIT_FAILED;
   }
   if(InpResetHalt) GlobalVariableDel(GV_HALT);
   if(!GlobalVariableCheck(GV_PEAK)) GlobalVariableSet(GV_PEAK, AccountInfoDouble(ACCOUNT_EQUITY));
   trade.SetExpertMagicNumber(InpMagic);
   EventSetTimer(1);
   Print("GoldGuardEA started. Peak equity=", GlobalVariableGet(GV_PEAK));
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason) { EventKillTimer(); }

void CloseAllBot(const string why)
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong t = PositionGetTicket(i);
      if(t == 0 || !PositionSelectByTicket(t)) continue;
      if(PositionGetInteger(POSITION_MAGIC) != InpMagic) continue;
      if(!trade.PositionClose(t)) Print("GoldGuardEA close failed ", t, " err=", GetLastError());
      else Print("GoldGuardEA closed ", t, ": ", why);
   }
}

void OnTimer()
{
   double eq   = AccountInfoDouble(ACCOUNT_EQUITY);
   double peak = GlobalVariableGet(GV_PEAK);
   if(eq > peak) { peak = eq; GlobalVariableSet(GV_PEAK, peak); }
   double dd = (peak > 0.0) ? (peak - eq) / peak * 100.0 : 0.0;

   if(dd >= InpMaxDrawdownPct && !GlobalVariableCheck(GV_HALT))
   {
      GlobalVariableSet(GV_HALT, 1.0);
      Alert("GoldGuardEA HARD KILL: drawdown ", DoubleToString(dd, 2), "% - closing all bot positions");
   }
   if(GlobalVariableCheck(GV_HALT)) { CloseAllBot("HALT active"); return; }

   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong t = PositionGetTicket(i);
      if(t == 0 || !PositionSelectByTicket(t)) continue;
      if(PositionGetInteger(POSITION_MAGIC) != InpMagic) continue;

      if(PositionGetDouble(POSITION_VOLUME) > InpMaxLot) { trade.PositionClose(t); Print("GoldGuardEA: lot above cap, closed ", t); continue; }

      if(PositionGetDouble(POSITION_SL) == 0.0)
      {
         string sym   = PositionGetString(POSITION_SYMBOL);
         double pt    = SymbolInfoDouble(sym, SYMBOL_POINT);
         double open  = PositionGetDouble(POSITION_PRICE_OPEN);
         bool   isBuy = (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY);
         double sl    = isBuy ? open - InpEmergencySLPts * pt : open + InpEmergencySLPts * pt;
         sl = NormalizeDouble(sl, (int)SymbolInfoInteger(sym, SYMBOL_DIGITS));
         if(trade.PositionModify(t, sl, PositionGetDouble(POSITION_TP)))
            Print("GoldGuardEA: emergency SL set on ", t, " at ", sl);
      }
   }
}
