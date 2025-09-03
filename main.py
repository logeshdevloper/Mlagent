import argparse
import sys
from config import Config
from src.data.binance_fetcher import BinanceFetcher
from src.api.supabase_client import SupabaseClient
from src.models.sequence_labeler import SequenceLabeler
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="Zone-Based Trading System (80% Accuracy Target)")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Collect command
    collect_parser = subparsers.add_parser('collect', help='Start data collection')
    collect_parser.add_argument('--symbol', default=Config.DEFAULT_SYMBOL, help='Trading symbol')
    collect_parser.add_argument('--max-iterations', type=int, help='Maximum iterations')
    
    # Label command
    label_parser = subparsers.add_parser('label', help='Generate labeled sequences')
    label_parser.add_argument('--symbol', default=Config.DEFAULT_SYMBOL, help='Trading symbol')
    label_parser.add_argument('--output', help='Output filename')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Check data status')
    status_parser.add_argument('--symbol', default=Config.DEFAULT_SYMBOL, help='Trading symbol')
    
    # Store past data command
    store_parser = subparsers.add_parser('store_past_data', help='Store historical data')
    store_parser.add_argument('--symbol', default=Config.DEFAULT_SYMBOL, help='Trading symbol')
    store_parser.add_argument('--days', type=int, default=7, help='Days to fetch')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test connections')
    
    # API command
    api_parser = subparsers.add_parser('api', help='Start Flask API server')
    api_parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    api_parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    api_parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Setup database tables')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'collect':
            fetcher = BinanceFetcher()
            fetcher.run_collector(args.symbol, args.max_iterations)
            
        elif args.command == 'label':
            labeler = SequenceLabeler()
            filepath = labeler.process_symbol(args.symbol, args.output)
            
            if filepath:
                print(f"✅ Feature engineering complete: {filepath}")
            else:
                print("❌ Feature engineering failed")
                sys.exit(1)
                
        elif args.command == 'status':
            db_client = SupabaseClient()
            count = db_client.get_candle_count(args.symbol)
            min_required = Config.SEQUENCE_LENGTH + Config.PREDICTION_HORIZON
            
            print(f"📊 {args.symbol}: {count:,} candles")
            if count >= min_required:
                print("✅ Ready for sequence generation")
            else:
                print(f"⏳ Need {min_required - count:,} more candles")

        elif args.command == 'store_past_data':
            from src.data.upload_past_data import fetch_binance_historical, upload_to_supabase
            
            print(f"📦 Fetching {args.days} days for {args.symbol}...")
            candles = fetch_binance_historical(args.symbol, args.days)
            upload_to_supabase(candles)
            print(f"✅ Stored {len(candles):,} candles")
                
        elif args.command == 'test':
            db_client = SupabaseClient()
            fetcher = BinanceFetcher()
            
            count = db_client.get_candle_count(Config.DEFAULT_SYMBOL)
            candle = fetcher.fetch_latest_candle()
            
            print(f"✅ Database: {count:,} candles")
            print(f"✅ Binance: Latest close ${candle['close']:,.2f}" if candle else "❌ Binance failed")
            
        elif args.command == 'api':
            from src.api.trading_api import app
            print(f"🚀 Starting Zone-Based Trading API on {args.host}:{args.port}")
            print("📊 Available endpoints:")
            print("  GET  / - Dashboard UI")
            print("  GET  /health - Health check")
            print("  GET  /predict_zones - Get zone predictions")
            print("  POST /session/start - Start 1-hour session")
            print("  GET  /session/update - Update session")
            print("  POST /session/end - End session")
            print("  GET  /zone_accuracy - Zone accuracy stats")
            print("  GET  /metrics - Model metrics")
            print(f"🔗 Dashboard: http://{args.host}:{args.port}")
            print(f"🔗 API Base: http://{args.host}:{args.port}/health")
            app.run(host=args.host, port=args.port, debug=args.debug)
            
        elif args.command == 'setup':
            db_client = SupabaseClient()
            print("📋 Zone Trading Tables SQL:")
            print(db_client.create_zone_tables_sql())
            print("\n💡 Run this SQL in your Supabase SQL editor to create all tables.")
            print("🔗 Go to: https://supabase.com/dashboard/project/[YOUR_PROJECT]/sql/new")
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()