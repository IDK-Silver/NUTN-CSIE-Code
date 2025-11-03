package com.example.user.ble_advertising;

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.AlertDialog;
import android.app.Dialog;
import android.bluetooth.BluetoothAdapter;
import android.bluetooth.BluetoothDevice;
import android.bluetooth.BluetoothManager;
import android.bluetooth.le.AdvertiseCallback;
import android.bluetooth.le.AdvertiseData;
import android.bluetooth.le.AdvertiseSettings;
import android.bluetooth.le.BluetoothLeAdvertiser;
import android.bluetooth.le.BluetoothLeScanner;
import android.bluetooth.le.ScanCallback;
import android.bluetooth.le.ScanRecord;
import android.bluetooth.le.ScanResult;
import android.bluetooth.le.ScanSettings;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.os.Handler;
import android.support.annotation.NonNull;
import android.support.annotation.Nullable;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.SparseArray;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ArrayAdapter;
import android.widget.LinearLayout;
import android.widget.ListAdapter;
import android.widget.ListView;
import android.widget.TableLayout;
import android.widget.TableRow;
import android.widget.TextView;
import android.widget.Toast;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;

import static java.lang.Math.abs;
import static java.lang.Math.*;

public class MainActivity extends AppCompatActivity {

    /*Bluetooth*/
    private BluetoothAdapter mBluetoothAdapter;
    public static final int GENERAL = 0xFFFF;

    /*Scanning*/
    private BluetoothLeScanner mBluetoothLeScanner;
    private boolean isScanning = false;
    private ScanCallback mScanCallback;
    private ArrayList<BluetoothDevice> mBluetoothDevices = new ArrayList<>();
    private ArrayList<String> scanResultList;
    private static final long SCAN_PERIOD = 60000;
    private Handler mHandler = new Handler();

    /*Advertising*/
    private BluetoothLeAdvertiser mBluetoothLeAdvertiser;
    private boolean isAdvertising = false;
    private AdvertiseCallback mAdvertiseCallback;

    /*Permission*/
    private static String[] PERMISSIONS_ACCESS = {
            Manifest.permission.ACCESS_FINE_LOCATION,
            Manifest.permission.ACCESS_COARSE_LOCATION,
            Manifest.permission.BLUETOOTH_SCAN,
            Manifest.permission.BLUETOOTH_ADVERTISE,
            Manifest.permission.BLUETOOTH_CONNECT};
    private static final int REQUEST_ACCESS_FINE_LOCATION = 1;
    private Dialog dialogPermission = null;
    private int REQUEST_ENABLE_BT = 1;

    /*UI*/
    private TextView textView_info;
    private ListView listView_scanResult;
    private TextView editText_data;
    private TableLayout scanResults = null;
    private final String[] kColors = {"#FFFFFF", "#FFE9BA"};
    private int colorIndex = 0;
    private int count = 0;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        checkBluetoothLowEnergyFeature();
        initBluetoothService();
        initScanAndAdvertiseCallback();
        initUI();

        scanResults = findViewById(R.id.table_scanResult);
    }

    /*onCreate checkBluetoothLowEnergyFeature*/
    private void checkBluetoothLowEnergyFeature() {
        if (!getPackageManager().hasSystemFeature(PackageManager.FEATURE_BLUETOOTH_LE)) {
            Toast.makeText(this, R.string.ble_not_supported, Toast.LENGTH_SHORT).show();
            finish();
        }
    }

    /*onCreate initBluetoothService*/
    private void initBluetoothService() {
        BluetoothManager bluetoothManager = (BluetoothManager) getSystemService(Context.BLUETOOTH_SERVICE);
        mBluetoothAdapter = bluetoothManager.getAdapter();
        if (mBluetoothAdapter == null) {
            Toast.makeText(this, R.string.bt_not_supported, Toast.LENGTH_SHORT).show();
            finish();
        }
    }

    /*onCreate initScanAndAdvertiseCallback*/
    private void initScanAndAdvertiseCallback() {
        mScanCallback = new ScanCallback() {
            @Override
            public void onScanResult(int callbackType, ScanResult result) {
                super.onScanResult(callbackType, result);
                mBluetoothDevices.clear();
                scanResultList.clear();
                saveScanResult(result);
                setAndUpdateListView();
            }

            @Override
            public void onBatchScanResults(List<ScanResult> results) {
                super.onBatchScanResults(results);
                mBluetoothDevices.clear();
                scanResultList.clear();
                for (ScanResult result : results) {
                    saveScanResult(result);
                }
                setAndUpdateListView();
            }

            @Override
            public void onScanFailed(int errorCode) {
                super.onScanFailed(errorCode);
                Toast.makeText(MainActivity.this
                        , "Error scanning devices: " + errorCode
                        , Toast.LENGTH_LONG).show();
            }
        };
        mAdvertiseCallback = new AdvertiseCallback() {
            @Override
            public void onStartSuccess(AdvertiseSettings settingsInEffect) {
                super.onStartSuccess(settingsInEffect);
            }

            @Override
            public void onStartFailure(int errorCode) {
                super.onStartFailure(errorCode);
            }
        };
    }

    /*onCreate initUI*/
    private void initUI() {
        textView_info = findViewById(R.id.textView_info);
        listView_scanResult = findViewById(R.id.listView_scanResult);
        scanResultList = new ArrayList<>();
        setAndUpdateListView();
        editText_data = findViewById(R.id.editText_data);
    }

    private void saveScanResult(ScanResult result) {
        if (result.getScanRecord() != null && hasManufacturerData(result.getScanRecord())) {
            String tempValue = unpackPayload(result.getScanRecord().getManufacturerSpecificData(GENERAL));
            tempValue = tempValue.substring(1, tempValue.length());

            if (!mBluetoothDevices.contains(result.getDevice())) {
                mBluetoothDevices.add(result.getDevice());

                double distance = pow(10, (abs(result.getRssi()) - 69) / (10.0 * 2));
                String distanceStr = String.format("%04.2f", distance);

                scanResultList.add("NUM_of_MSG : " + count + System.getProperty("line.separator") +
                        "Address : " + result.getDevice().getAddress() + System.getProperty("line.separator") +
                        "RSSI : " + result.getRssi() + System.getProperty("line.separator") +
                        "Distance : " + distanceStr + " m" + System.getProperty("line.separator") +
                        "MSG : " + tempValue);
                count = count + 1;
            }
        }
    }

    private boolean hasManufacturerData(ScanRecord record) {
        SparseArray<byte[]> data = record.getManufacturerSpecificData();
        return (data != null && data.get(GENERAL) != null);
    }

    private String unpackPayload(byte[] data) {
        ByteBuffer buffer = ByteBuffer.wrap(data)
                .order(ByteOrder.LITTLE_ENDIAN);
        buffer.get();
        byte[] b = new byte[buffer.limit()];
        for (int i = 0; i < buffer.limit(); i++) {
            b[i] = buffer.get(i);
        }
        try {
            return (new String(b, "UTF-8"));
        } catch (Exception e) {
            return " Unable to unpack.";
        }
    }

    private void setAndUpdateListView() {
        for (String scanResult : scanResultList) {
            TableRow row = new TableRow(this);
            TextView text = new TextView(this);

            text.setText(scanResult);
            text.setBackgroundColor(Color.parseColor(getColor()));

            text.setPadding(0, 0, 0, 150);
            row.addView(text);

            scanResults.addView(row, 0);
        }
    }

    private String getColor() {
        String color = kColors[colorIndex];
        ++colorIndex;
        if (colorIndex >= kColors.length) {
            colorIndex = 0;
        }
        return color;
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (android.os.Build.VERSION.SDK_INT >= 23) {
            checkPermission();
        } else {
            checkBluetoothEnableThenScanAndAdvertising();
        }
    }

    private void checkPermission() {
        int permission = ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION);
        if (permission == PackageManager.PERMISSION_GRANTED) {
            checkBluetoothEnableThenScanAndAdvertising();
        } else {
            showDialogForPermission();
        }
    }

    private void showDialogForPermission() {
        AlertDialog.Builder dialogBuilder = new AlertDialog.Builder(MainActivity.this);
        dialogBuilder.setTitle(getResources().getString(R.string.dialog_permission_title));
        dialogBuilder.setMessage(getResources().getString(R.string.dialog_permission_message));
        dialogBuilder.setPositiveButton(getResources().getString(R.string.dialog_permission_ok)
                , new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialogInterface, int i) {
                        ActivityCompat.requestPermissions(MainActivity.this
                                , PERMISSIONS_ACCESS
                                , REQUEST_ACCESS_FINE_LOCATION);
                    }
                });
        dialogBuilder.setNegativeButton(getResources().getString(R.string.dialog_permission_no)
                , new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialogInterface, int i) {
                        Toast.makeText(MainActivity.this
                                , getResources().getString(R.string.dialog_permission_toast_negative)
                                , Toast.LENGTH_LONG).show();
                    }
                });
        if (dialogPermission == null) {
            dialogPermission = dialogBuilder.create();
        }
        if (!dialogPermission.isShowing()) {
            dialogPermission.show();
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        switch (requestCode) {
            case REQUEST_ACCESS_FINE_LOCATION: {
                if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    checkBluetoothEnableThenScanAndAdvertising();
                } else {
                    Toast.makeText(MainActivity.this
                            , getResources().getString(R.string.dialog_permission_toast_negative)
                            , Toast.LENGTH_LONG).show();
                }
                break;
            }
        }
    }

    private void checkBluetoothEnableThenScanAndAdvertising() {
        if (mBluetoothAdapter.isEnabled()) {
            startScanLeDevice();
            startAdvertising();
        } else {
            openBluetoothSetting();
        }
    }

    @SuppressLint("MissingPermission")
    private void startScanLeDevice() {
        if (isScanning) {
            return;
        }
        mHandler.postDelayed(new Runnable() {
            @Override
            public void run() {
                stopScanLeDevice();
            }
        }, SCAN_PERIOD);

        isScanning = true;
        mBluetoothLeScanner = mBluetoothAdapter.getBluetoothLeScanner();
        int reportDelay = 0;
        if (mBluetoothAdapter.isOffloadedScanBatchingSupported()) {
            reportDelay = 1000;
        }
        ScanSettings settings = new ScanSettings.Builder()
                .setScanMode(ScanSettings.SCAN_MODE_LOW_LATENCY)
                .setReportDelay(reportDelay)
                .build();
        mBluetoothLeScanner.startScan(null, settings, mScanCallback);
        textView_info.setText(getResources().getString(R.string.bt_scanning));
    }

    @SuppressLint("MissingPermission")
    private void stopScanLeDevice() {
        if (isScanning) {
            mBluetoothLeScanner.stopScan(mScanCallback);
            isScanning = false;
            textView_info.setText(getResources().getString(R.string.bt_stop_scan));
        }
    }

    @SuppressLint("MissingPermission")
    private void startAdvertising() {
        if (isAdvertising) {
            return;
        }
        isAdvertising = true;
        mBluetoothLeAdvertiser = mBluetoothAdapter.getBluetoothLeAdvertiser();
        AdvertiseSettings settings = new AdvertiseSettings.Builder()
                .setAdvertiseMode(AdvertiseSettings.ADVERTISE_MODE_LOW_LATENCY)
                .setConnectable(false)
                .setTimeout(0)
                .setTxPowerLevel(AdvertiseSettings.ADVERTISE_TX_POWER_HIGH)
                .build();
        AdvertiseData data = new AdvertiseData.Builder()
                .addManufacturerData(GENERAL, buildPayload(editText_data.getText().toString()))
                .build();
        mBluetoothLeAdvertiser.startAdvertising(settings, data, mAdvertiseCallback);
    }

    private byte[] buildPayload(String value) {
        byte flags = (byte) 0x8000000;
        byte[] b = {};
        try {
            b = value.getBytes("UTF-8");
        } catch (Exception e) {
            return b;
        }
        int max = 26;//如果加device name最大是16個字，不加是26(不含flag)
        int capacity;
        if (b.length <= max) {
            capacity = b.length + 1;
        } else {
            capacity = max + 1;
            System.arraycopy(b, 0, b, 0, max);
        }
        byte[] output;
        output = ByteBuffer.allocate(capacity)
                .order(ByteOrder.LITTLE_ENDIAN) //GATT APIs expect LE order
                .put(flags) //Add the flags byte
                .put(b)
                .array();
        return output;
    }

    @SuppressLint("MissingPermission")
    private void openBluetoothSetting() {
        Intent bluetoothSettingIntent = new Intent(BluetoothAdapter.ACTION_REQUEST_ENABLE);
        startActivityForResult(bluetoothSettingIntent, REQUEST_ENABLE_BT);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == REQUEST_ENABLE_BT) {
            checkBluetoothEnableThenScanAndAdvertising();
        }
    }

    public void btnClick(View v) {
        int id = v.getId();
        if (id == R.id.button_scan) {
            startScanLeDevice();
        } else if (id == R.id.button_stop) {
            stopScanLeDevice();
        } else if (id == R.id.button_save) {
            stopAndRestartAdvertising();
        }
    }

    @SuppressLint("MissingPermission")
    private void stopAndRestartAdvertising() {
        if (isAdvertising) {
            mBluetoothLeAdvertiser = mBluetoothAdapter.getBluetoothLeAdvertiser();
            mBluetoothLeAdvertiser.stopAdvertising(mAdvertiseCallback);
            isAdvertising = false;
        }
        startAdvertising();
    }

}
