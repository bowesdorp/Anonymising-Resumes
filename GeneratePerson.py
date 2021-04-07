from faker import Faker

if __name__ == '__main__':
    fake = Faker()
    f_name = fake.name()
    f_address = fake.address()
    f_phone = fake.phone_number()
    f_dob = fake.date_of_birth()
    f_email = fake.email()
    f_country = fake.country()

    print(f'name: {f_name}')
    print(f'address: {f_address}')
    print(f'phone: {f_phone}')
    print(f'dob: {f_dob}')
    print(f'email: {f_email}')
    print(f'nationality: {f_country}')