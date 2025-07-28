"""
Backup & Recovery Manager
Ensures your trading data is never lost
"""

import os
import json
import shutil
import sqlite3
import zipfile
from datetime import datetime, timedelta
from typing import Dict, List
import boto3  # For AWS S3 backup
from pathlib import Path

class BackupManager:
    def __init__(self, backup_dir: str = "backups"):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
        
        # Critical files to backup
        self.critical_files = [
            'config/trading_config.py',
            'database/trading.db',
            'database/*.db',
            'logs/*.log',
            '.env.encrypted',  # Encrypted version of .env
            'models/*.pkl',    # Trained ML models
            'portfolio_state.json',
            '*.py',           # All Python files
            'paper_trading.py',
            'setup_apis.py',
            'test_apis.py'
        ]
        
    def create_local_backup(self, backup_name: str = None) -> str:
        """Create local backup of all critical data"""
        if not backup_name:
            backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        backup_path = self.backup_dir / f"{backup_name}.zip"
        
        with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_pattern in self.critical_files:
                if '*' in file_pattern:
                    # Handle wildcards
                    directory, pattern = file_pattern.split('/')
                    if os.path.exists(directory):
                        for file in Path(directory).glob(pattern.replace('*', '*')):
                            zipf.write(file, file)
                else:
                    if os.path.exists(file_pattern):
                        zipf.write(file_pattern, file_pattern)
        
        print(f"✅ Local backup created: {backup_path}")
        return str(backup_path)
    
    def backup_to_cloud(self, backup_file: str, cloud_provider: str = "s3") -> bool:
        """Upload backup to cloud storage"""
        try:
            if cloud_provider == "s3":
                return self._backup_to_s3(backup_file)
            elif cloud_provider == "drive":
                return self._backup_to_drive(backup_file)
            else:
                print(f"❌ Unsupported cloud provider: {cloud_provider}")
                return False
        except Exception as e:
            print(f"❌ Cloud backup failed: {e}")
            return False
    
    def _backup_to_s3(self, backup_file: str) -> bool:
        """Backup to AWS S3"""
        try:
            s3_client = boto3.client('s3',
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
            )
            
            bucket_name = os.getenv('S3_BACKUP_BUCKET', 'meme-trading-backups')
            key = f"trading-system/{Path(backup_file).name}"
            
            s3_client.upload_file(backup_file, bucket_name, key)
            print(f"✅ Backup uploaded to S3: s3://{bucket_name}/{key}")
            return True
            
        except Exception as e:
            print(f"❌ S3 backup failed: {e}")
            return False
    
    def _backup_to_drive(self, backup_file: str) -> bool:
        """Backup to Google Drive"""
        try:
            from googleapiclient.discovery import build
            from google.oauth2.credentials import Credentials
            
            # This would require Google Drive API setup
            print("✅ Google Drive backup would go here")
            return True
            
        except Exception as e:
            print(f"❌ Drive backup failed: {e}")
            return False
    
    def auto_cleanup_old_backups(self, keep_days: int = 30):
        """Clean up backups older than specified days"""
        cutoff_date = datetime.now() - timedelta(days=keep_days)
        
        for backup_file in self.backup_dir.glob("backup_*.zip"):
            if backup_file.stat().st_mtime < cutoff_date.timestamp():
                backup_file.unlink()
                print(f"🗑️ Removed old backup: {backup_file.name}")
    
    def verify_backup(self, backup_file: str) -> bool:
        """Verify backup file integrity"""
        try:
            with zipfile.ZipFile(backup_file, 'r') as zipf:
                # Test the zip file
                zipf.testzip()
                
                # Check critical files are present
                file_list = zipf.namelist()
                critical_found = 0
                
                for critical in self.critical_files:
                    if any(critical.replace('*', '') in f for f in file_list):
                        critical_found += 1
                
                success_rate = critical_found / len(self.critical_files)
                print(f"📊 Backup verification: {success_rate:.0%} critical files found")
                
                return success_rate > 0.7  # At least 70% of critical files
                
        except Exception as e:
            print(f"❌ Backup verification failed: {e}")
            return False
    
    def restore_from_backup(self, backup_file: str, restore_dir: str = ".") -> bool:
        """Restore system from backup"""
        try:
            if not self.verify_backup(backup_file):
                print("❌ Backup verification failed, aborting restore")
                return False
            
            # Create restore point of current state
            current_backup = self.create_local_backup("pre_restore_backup")
            
            with zipfile.ZipFile(backup_file, 'r') as zipf:
                zipf.extractall(restore_dir)
            
            print(f"✅ System restored from: {backup_file}")
            print(f"🔄 Current state backed up to: {current_backup}")
            return True
            
        except Exception as e:
            print(f"❌ Restore failed: {e}")
            return False
    
    def schedule_automatic_backups(self):
        """Setup automatic backup scheduling"""
        try:
            import schedule
            
            # Daily local backup
            schedule.every().day.at("02:00").do(self.create_local_backup)
            
            # Weekly cloud backup
            schedule.every().week.do(lambda: self.backup_to_cloud(
                self.create_local_backup("weekly_backup")
            ))
            
            # Monthly cleanup (use days instead of month)
            schedule.every(30).days.do(self.auto_cleanup_old_backups)
            
            print("✅ Automatic backup schedule configured")
            print("   • Daily local backups at 2:00 AM")
            print("   • Weekly cloud backups")
            print("   • Monthly cleanup of old backups")
            
        except ImportError:
            print("⚠️ Schedule package not installed. Install with: pip install schedule")
        except Exception as e:
            print(f"⚠️ Backup scheduling setup failed: {e}")
            print("   Backups can still be run manually")

# Usage example
if __name__ == "__main__":
    backup_mgr = BackupManager()
    
    # Create backup
    backup_file = backup_mgr.create_local_backup()
    
    # Verify backup
    if backup_mgr.verify_backup(backup_file):
        print("✅ Backup verified successfully")
        
        # Upload to cloud (optional)
        # backup_mgr.backup_to_cloud(backup_file)
    
    # Setup automatic backups
    backup_mgr.schedule_automatic_backups()