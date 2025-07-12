import hashlib

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Generate hashes for your passwords
passwords = ["password1", "password2"] #don't bother trying, that's not my real password
for pwd in passwords:
    print(f"Password: {pwd}")
    print(f"Hash: {hash_password(pwd)}")
    print("-" * 40)