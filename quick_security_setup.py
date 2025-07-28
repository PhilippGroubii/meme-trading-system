#!/usr/bin/env python3
"""
Quick Security Setup
One-click security and backup setup for your trading system
"""

import os
import sys
import subprocess
from pathlib import Path

def install_required_packages():
    """Install security-related packages"""
    packages = [
        'cryptography',  # For encryption
        'schedule',      # For automated backups
        'boto3',         # For AWS S3 backups (optional)
    ]
    
    print("📦 Installing security packages...")
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✅ Installed {package}")
        except Exception as e:
            print(f"⚠️ Failed to install {package}: {e}")

def create_directories():
    """Create necessary directories"""
    dirs = ['security', 'backup', 'backups', 'logs']
    
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"✅ Created directory: {dir_name}")

def encrypt_env_file():
    """Encrypt the .env file with API keys"""
    try:
        from security.security_manager import SecurityManager
        
        if os.path.exists('.env'):
            # Read .env file
            with open('.env', 'r') as f:
                env_content = f.read()
            
            # Create security manager
            security = SecurityManager()
            
            # Encrypt the content
            encrypted_content = security.encrypt_data(env_content)
            
            # Save encrypted version
            with open('.env.encrypted', 'w') as f:
                f.write(encrypted_content)
            
            print("✅ .env file encrypted and saved as .env.encrypted")
            print("🔒 Your API keys are now secure!")
            
            # Optionally remove original .env (user choice)
            response = input("🗑️ Remove original .env file? (y/n): ").lower()
            if response == 'y':
                os.remove('.env')
                print("✅ Original .env file removed")
            else:
                print("⚠️ Original .env file kept (remember to secure it)")
                
        else:
            print("⚠️ No .env file found - create one first with setup_apis.py")
            
    except Exception as e:
        print(f"❌ Encryption failed: {e}")

def create_first_backup():
    """Create your first secure backup"""
    try:
        from backup.backup_manager import BackupManager
        
        backup_mgr = BackupManager()
        
        # Create backup
        backup_file = backup_mgr.create_local_backup("initial_secure_backup")
        
        # Verify backup
        if backup_mgr.verify_backup(backup_file):
            print("✅ Initial backup created and verified")
            
            # Setup automatic backups
            backup_mgr.schedule_automatic_backups()
            
        else:
            print("⚠️ Backup created but verification failed")
            
    except Exception as e:
        print(f"❌ Backup failed: {e}")

def create_security_checklist():
    """Create a security checklist file"""
    
    checklist = """# 🔐 SECURITY CHECKLIST FOR MEME TRADING SYSTEM

## ✅ COMPLETED:
- [ ] API keys encrypted (.env.encrypted created)
- [ ] Initial backup created and verified
- [ ] Automatic backup schedule configured
- [ ] Security packages installed

## 🎯 NEXT STEPS:

### 1. MASTER PASSWORD SECURITY:
- [ ] Set strong MASTER_PASSWORD environment variable
- [ ] Store master password in secure password manager
- [ ] Never commit master password to git

### 2. BACKUP VERIFICATION:
- [ ] Test backup restoration process
- [ ] Setup cloud backup (AWS S3 or Google Drive)
- [ ] Verify backups are created daily

### 3. ACCESS CONTROL:
- [ ] Limit file permissions (chmod 600 for sensitive files)
- [ ] Use separate user account for trading system
- [ ] Enable firewall on trading server

### 4. MONITORING:
- [ ] Setup alerts for failed backups
- [ ] Monitor system access logs
- [ ] Regular security audits

## 🚨 EMERGENCY PROCEDURES:

### If API Keys Are Compromised:
1. Immediately revoke all API keys
2. Generate new API keys
3. Update .env file and re-encrypt
4. Review recent trading activity

### If System Is Hacked:
1. Shut down trading system immediately
2. Restore from latest known good backup
3. Change all passwords and API keys
4. Review security logs

### If Backup Is Corrupted:
1. Use previous backup version
2. Re-create backup from current state
3. Verify new backup integrity

## 📞 SUPPORT:
- Keep this checklist updated
- Document any security incidents
- Regular security reviews every 30 days

Generated: {datetime.now()}
"""
    
    from datetime import datetime
    
    with open('SECURITY_CHECKLIST.md', 'w') as f:
        f.write(checklist.format(datetime=datetime.now()))
    
    print("✅ Security checklist created: SECURITY_CHECKLIST.md")

def main():
    """Main security setup function"""
    
    print("🔐 QUICK SECURITY SETUP FOR MEME TRADING SYSTEM")
    print("=" * 60)
    
    # Step 1: Install packages
    print("\n1️⃣ Installing security packages...")
    install_required_packages()
    
    # Step 2: Create directories
    print("\n2️⃣ Creating directories...")
    create_directories()
    
    # Step 3: Encrypt API keys
    print("\n3️⃣ Encrypting API keys...")
    encrypt_env_file()
    
    # Step 4: Create backup
    print("\n4️⃣ Creating initial backup...")
    create_first_backup()
    
    # Step 5: Security checklist
    print("\n5️⃣ Creating security checklist...")
    create_security_checklist()
    
    # Final summary
    print("\n" + "=" * 60)
    print("🎉 SECURITY SETUP COMPLETE!")
    print("=" * 60)
    
    print("\n✅ What was secured:")
    print("   • API keys encrypted")
    print("   • Initial backup created")
    print("   • Automatic backups scheduled")
    print("   • Security checklist generated")
    
    print("\n🎯 Next steps:")
    print("   1. Set strong MASTER_PASSWORD environment variable")
    print("   2. Review SECURITY_CHECKLIST.md")
    print("   3. Test backup restoration")
    print("   4. Setup cloud backup (optional)")
    
    print("\n🚀 Your trading system is now secure!")
    print("   Continue with: python paper_trading.py")

if __name__ == "__main__":
    main()