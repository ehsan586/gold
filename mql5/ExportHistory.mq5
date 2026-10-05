//+------------------------------------------------------------------+
//| ExportHistory.mq5 - script: exports M1 candles to CSV for the    |
//| Python backtester.  Output: MQL5\Files\<InpFile>                  |
//| Columns: time(epoch),open,high,low,close,tick_volume,spread       |
//| NOT COMPILED/TESTED by the developer: compile in MetaEditor (F7). |
//| Tip: Tools > Options > Charts > "Max bars in chart" = Unlimited,  |
//| and scroll the M1 chart back in time first so history is loaded.  |
//+------------------------------------------------------------------+
#property script_show_inputs
input string InpSymbol = "";               // empty = current chart symbol
input int    InpBars   = 200000;           // number of M1 bars
input string InpFile   = "XAUUSD_M1.csv";

void OnStart()
{
   string sym = (InpSymbol == "") ? _Symbol : InpSymbol;
   MqlRates rates[];
   ArraySetAsSeries(rates, false);          // oldest first
   int n = CopyRates(sym, PERIOD_M1, 0, InpBars, rates);
   if(n <= 0) { Print("CopyRates failed, err=", GetLastError()); return; }
   int h = FileOpen(InpFile, FILE_WRITE | FILE_CSV | FILE_ANSI, ',');
   if(h == INVALID_HANDLE) { Print("FileOpen failed, err=", GetLastError()); return; }
   FileWrite(h, "time", "open", "high", "low", "close", "tick_volume", "spread");
   for(int i = 0; i < n; i++)
      FileWrite(h, (long)rates[i].time,
                DoubleToString(rates[i].open,  _Digits), DoubleToString(rates[i].high, _Digits),
                DoubleToString(rates[i].low,   _Digits), DoubleToString(rates[i].close, _Digits),
                (long)rates[i].tick_volume, rates[i].spread);
   FileClose(h);
   Print("Exported ", n, " bars of ", sym, " to ", TerminalInfoString(TERMINAL_DATA_PATH), "\\MQL5\\Files\\", InpFile);
}
