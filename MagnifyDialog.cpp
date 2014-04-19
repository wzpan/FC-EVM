// Yet anther C++ implementation of EVM, based on OpenCV and Qt. 
// Copyright (C) 2014  Joseph Pan <cs.wzpan@gmail.com>
// 
// This library is free software; you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as
// published by the Free Software Foundation; either version 2.1 of the
// License, or (at your option) any later version.
// 
// This library is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// 
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
// 02110-1301 USA
// 

#include "MagnifyDialog.h"
#include <string.h>
#include "ui_MagnifyDialog.h"

MagnifyDialog::MagnifyDialog(QWidget *parent, VideoProcessor *processor) :
    QDialog(parent),
    ui(new Ui::MagnifyDialog)
{
    ui->setupUi(this);

    this->processor = processor;

    alphaStr = ui->alphaLabel->text();
    lambdaStr = ui->lambdaLabel->text();
    flStr = ui->flLabel->text();
    fhStr = ui->fhLabel->text();
    chromStr = ui->chromLabel->text();
    vminStr = ui->vminLabel->text();
    vmaxStr = ui->vmaxLabel->text();
    sminStr = ui->sminLabel->text();

    std::stringstream ss;
    ss << alphaStr.toStdString() << processor->alpha;
    ui->alphaLabel->setText(QString::fromStdString(ss.str()));
    ss.str("");
    ss << lambdaStr.toStdString() << processor->lambda_c;
    ui->lambdaLabel->setText(QString::fromStdString(ss.str()));
    ss.str("");
    ss << flStr.toStdString() << processor->fl;
    ui->flLabel->setText(QString::fromStdString(ss.str()));
    ss.str("");
    ss << fhStr.toStdString() << processor->fh;
    ui->fhLabel->setText(QString::fromStdString(ss.str()));
    ss.str("");
    ss << chromStr.toStdString() << processor->chromAttenuation;
    ui->chromLabel->setText(QString::fromStdString(ss.str()));
    ss.str("");
    ss << vminStr.toStdString() << processor->vmin;
    ui->vminLabel->setText(QString::fromStdString(ss.str()));
    ss.str("");
    ss << vmaxStr.toStdString() << processor->vmax;
    ui->vmaxLabel->setText(QString::fromStdString(ss.str()));
    ss.str("");
    ss << sminStr.toStdString() << processor->smin;
    ui->sminLabel->setText(QString::fromStdString(ss.str()));
    ss.str("");

    ui->vminSlider->setEnabled(processor->isLabeled);
    ui->vmaxSlider->setEnabled(processor->isLabeled);
    ui->sminSlider->setEnabled(processor->isLabeled);
    ui->checkGrabcut->setEnabled(processor->isLabeled);
    ui->checkTrack->setEnabled(processor->isLabeled);
    ui->checkKalman->setEnabled(processor->isLabeled);
}

MagnifyDialog::~MagnifyDialog()
{
    delete ui;
}

void MagnifyDialog::enableCheck(bool enabled)
{
    ui->checkTrack->setEnabled(enabled);
    ui->checkGrabcut->setEnabled(enabled);
    ui->checkKalman->setEnabled(enabled);

    ui->checkTrack->setChecked(enabled);
    ui->checkGrabcut->setChecked(enabled);
    ui->checkKalman->setChecked(enabled);

    processor->isTrack = enabled;
    processor->isGrabcut = enabled;
    processor->isKalman = enabled;

    ui->vminSlider->setEnabled(enabled);
    ui->vmaxSlider->setEnabled(enabled);
    ui->sminSlider->setEnabled(enabled);
}

void MagnifyDialog::on_alphaSlider_valueChanged(int value)
{
    processor->alpha = value;
    std::stringstream ss;
    ss << alphaStr.toStdString() << processor->alpha;
    ui->alphaLabel->setText(QString::fromStdString(ss.str()));
}

void MagnifyDialog::on_lambdaSlider_valueChanged(int value)
{
    processor->lambda_c = value;
    std::stringstream ss;
    ss << lambdaStr.toStdString() << processor->lambda_c;
    ui->lambdaLabel->setText(QString::fromStdString(ss.str()));
}

void MagnifyDialog::on_flSlider_valueChanged(int value)
{
    processor->fl = value / 100.0;
    std::stringstream ss;
    ss << flStr.toStdString() << processor->fl;
    ui->flLabel->setText(QString::fromStdString(ss.str()));
}

void MagnifyDialog::on_fhSlider_valueChanged(int value)
{
    processor->fh = value / 100.0;
    std::stringstream ss;
    ss << fhStr.toStdString() << processor->fh;
    ui->fhLabel->setText(QString::fromStdString(ss.str()));
}

void MagnifyDialog::on_chromSlider_valueChanged(int value)
{
    processor->chromAttenuation = value / 10.0;
    std::stringstream ss;
    ss << chromStr.toStdString() << processor->chromAttenuation;
    ui->chromLabel->setText(QString::fromStdString(ss.str()));
}

void MagnifyDialog::on_checkTrack_toggled(bool checked)
{
    processor->isTrack = checked;
    if (checked)
        ui->checkGrabcut->setEnabled(true);
    else
        ui->checkGrabcut->setEnabled(false);
}

void MagnifyDialog::on_checkGrabcut_toggled(bool checked)
{
    processor->isGrabcut = checked;
}

void MagnifyDialog::on_checkKalman_toggled(bool checked)
{
    processor->isKalman = checked;
}

void MagnifyDialog::on_vminSlider_valueChanged(int value)
{
    processor->vmin = value;
    std::stringstream ss;
    ss << vminStr.toStdString() << processor->vmin;
    ui->vminLabel->setText(QString::fromStdString(ss.str()));
}

void MagnifyDialog::on_vmaxSlider_valueChanged(int value)
{
    processor->vmax = value;
    std::stringstream ss;
    ss << vmaxStr.toStdString() << processor->vmax;
    ui->vmaxLabel->setText(QString::fromStdString(ss.str()));
}

void MagnifyDialog::on_sminSlider_valueChanged(int value)
{
    processor->smin = value;
    std::stringstream ss;
    ss << sminStr.toStdString() << processor->smin;
    ui->sminLabel->setText(QString::fromStdString(ss.str()));
}
