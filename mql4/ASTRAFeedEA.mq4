//+------------------------------------------------------------------+
//| ASTRAFeedEA.mq4 - READ-ONLY market data bridge for ASTRA Trader  |
//| Writes tick/account/symbol metadata + complete closed TF files.  |
//| It NEVER sends, modifies, or closes orders.                       |
//+------------------------------------------------------------------+
#property strict
#property version "1.00"

input int InpBars = 400;
input int InpRefreshSeconds = 2;
input string InpSymbol = "";

string Sym() { return (InpSymbol == "" ? Symbol() : InpSymbol); }

void WriteMeta()
{
   string sym = Sym();
   RefreshRates();
   int digits = (int)MarketInfo(sym, MODE_DIGITS);
   double bid = MarketInfo(sym, MODE_BID);
   double ask = MarketInfo(sym, MODE_ASK);
   if(bid <= 0 || ask <= 0) return;

   int h = FileOpen("ASTRA_MT4_FEED.json", FILE_WRITE|FILE_TXT|FILE_ANSI|FILE_COMMON);
   if(h == INVALID_HANDLE) { Print("ASTRAFeedEA JSON FileOpen failed: ", GetLastError()); return; }
   FileWriteString(h, "{\n");
   FileWriteString(h, "\"updated_at\":" + DoubleToString(TimeCurrent(),0) + ",\n");
   FileWriteString(h, "\"symbol\":\"" + sym + "\",\n");
   FileWriteString(h, "\"tick\":{\"time\":" + DoubleToString(TimeCurrent(),0) + ",\"bid\":" + DoubleToString(bid,digits) + ",\"ask\":" + DoubleToString(ask,digits) + "},\n");
   double margin = AccountMargin();
   double margin_level = 0.0;
   if(margin > 0.0) margin_level = (AccountEquity() / margin) * 100.0;
   FileWriteString(h, "\"account\":{\"login\":" + IntegerToString(AccountNumber()) +
                    ",\"balance\":" + DoubleToString(AccountBalance(),2) +
                    ",\"equity\":" + DoubleToString(AccountEquity(),2) +
                    ",\"margin\":" + DoubleToString(margin,2) +
                    ",\"free_margin\":" + DoubleToString(AccountFreeMargin(),2) +
                    ",\"margin_level\":" + DoubleToString(margin_level,2) +
                    ",\"leverage\":" + IntegerToString(AccountLeverage()) +
                    ",\"margin_mode\":\"HEDGING\",\"is_demo\":" + (IsDemo() ? "true" : "false") +
                    ",\"currency\":\"" + AccountCurrency() + "\"},\n");
   FileWriteString(h, "\"symbol_spec\":{\"digits\":" + IntegerToString(digits) +
                    ",\"point\":" + DoubleToString(MarketInfo(sym,MODE_POINT),10) +
                    ",\"tick_size\":" + DoubleToString(MarketInfo(sym,MODE_TICKSIZE),10) +
                    ",\"tick_value\":" + DoubleToString(MarketInfo(sym,MODE_TICKVALUE),10) +
                    ",\"contract_size\":" + DoubleToString(MarketInfo(sym,MODE_LOTSIZE),4) +
                    ",\"volume_min\":" + DoubleToString(MarketInfo(sym,MODE_MINLOT),4) +
                    ",\"volume_max\":" + DoubleToString(MarketInfo(sym,MODE_MAXLOT),4) +
                    ",\"volume_step\":" + DoubleToString(MarketInfo(sym,MODE_LOTSTEP),4) +
                    ",\"stops_level\":" + IntegerToString((int)MarketInfo(sym,MODE_STOPLEVEL)) +
                    ",\"freeze_level\":0,\"trade_allowed\":false,\"currency_profit\":\"" + AccountCurrency() + "\"},\n");
   FileWriteString(h, "\"read_only\":true\n}");
   FileClose(h);
}

void WriteHistoryTF(string label, int tf)
{
   string sym = Sym();
   int digits = (int)MarketInfo(sym, MODE_DIGITS);
   string file = "ASTRA_MT4_" + label + ".csv";
   int c = FileOpen(file, FILE_WRITE|FILE_CSV|FILE_ANSI|FILE_COMMON, ',');
   if(c == INVALID_HANDLE) { Print("ASTRAFeedEA history FileOpen failed: ", label, " err=", GetLastError()); return; }
   FileWrite(c,"time","open","high","low","close","volume");
   int total = iBars(sym, tf);
   int count = MathMin(InpBars, total);
   // skip the currently-forming bar: start at shift 1
   for(int i=count; i>=1; i--)
      FileWrite(c, (long)iTime(sym,tf,i), DoubleToString(iOpen(sym,tf,i),digits),
                DoubleToString(iHigh(sym,tf,i),digits), DoubleToString(iLow(sym,tf,i),digits),
                DoubleToString(iClose(sym,tf,i),digits), (long)iVolume(sym,tf,i));
   FileClose(c);
}

void WriteFeed()
{
   WriteMeta();
   WriteHistoryTF("M1", PERIOD_M1);
   WriteHistoryTF("M5", PERIOD_M5);
   WriteHistoryTF("M15", PERIOD_M15);
   WriteHistoryTF("M30", PERIOD_M30);
   WriteHistoryTF("H1", PERIOD_H1);
   WriteHistoryTF("H4", PERIOD_H4);
}

int OnInit()
{
   if(!IsDemo()) { Alert("ASTRAFeedEA is read-only but demo-only by default. Attach it to a DEMO terminal."); return(INIT_FAILED); }
   EventSetTimer(MathMax(1,InpRefreshSeconds));
   WriteFeed();
   Print("ASTRAFeedEA started. FILE_COMMON feed is read-only. No orders are sent.");
   return(INIT_SUCCEEDED);
}
void OnDeinit(const int reason) { EventKillTimer(); }
void OnTimer() { WriteFeed(); }
void OnTick() { WriteMeta(); }
