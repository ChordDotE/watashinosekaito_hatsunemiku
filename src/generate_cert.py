from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
import datetime
import os
from pathlib import Path
import sys

# venvディレクトリのパスを取得
venv_dir = Path(__file__).parent

# utils.path_configをインポート
sys.path.append(str(venv_dir))
from utils.path_config import PathConfig

# PathConfigを初期化
path_config = PathConfig.initialize(venv_dir)

# 証明書と鍵のパスを取得
cert_dir = path_config.certs_dir
cert_path = path_config.cert_file
key_path = path_config.key_file

# ディレクトリが存在することを確認
os.makedirs(cert_dir, exist_ok=True)

# 秘密鍵の生成
key = rsa.generate_private_key(
    public_exponent=65537,
    key_size=2048,
)

# 証明書の詳細情報
subject = issuer = x509.Name([
    x509.NameAttribute(NameOID.COUNTRY_NAME, u"JP"),
    x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, u"Tokyo"),
    x509.NameAttribute(NameOID.LOCALITY_NAME, u"Shibuya"),
    x509.NameAttribute(NameOID.ORGANIZATION_NAME, u"Development"),
    x509.NameAttribute(NameOID.COMMON_NAME, u"localhost"),
])

# 証明書の生成
cert = x509.CertificateBuilder().subject_name(
    subject
).issuer_name(
    issuer
).public_key(
    key.public_key()
).serial_number(
    x509.random_serial_number()
).not_valid_before(
    datetime.datetime.utcnow()
).not_valid_after(
    # 証明書の有効期限（1年間）
    datetime.datetime.utcnow() + datetime.timedelta(days=365)
).add_extension(
    x509.SubjectAlternativeName([x509.DNSName(u"localhost")]),
    critical=False,
).sign(key, hashes.SHA256())

# 秘密鍵をPEM形式で保存
with open(key_path, "wb") as f:
    f.write(key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    ))

# 証明書をPEM形式で保存
with open(cert_path, "wb") as f:
    f.write(cert.public_bytes(serialization.Encoding.PEM))

print(f"証明書と秘密鍵が生成されました。")
print(f"証明書のパス: {cert_path}")
print(f"秘密鍵のパス: {key_path}")
